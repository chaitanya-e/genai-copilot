import json, time, httpx, difflib
from pathlib import Path

from mlops.mlflow.tracking import log_eval_run

# Loads a list of test questions from a file (quests.json1).
# Sends each question to your API endpoint /ask.
# Measures response time.
# Compares the API’s answer with an expected hint using similarity ratio.
# Prints results + average latency

QUEST_PATH = Path(__file__).with_name("quests.json1")
API = "http://localhost:8000/ask"

# Create default quests.json1 if it doesn’t exist
if not QUEST_PATH.exists():
    QUEST_PATH.write_text("\n".join([
        json.dumps({"question":"what is this project and how do i query it?", "hint":"FastAPI, /ask endpoint"}),
        ]), encoding="utf-8")
    
qs = [json.loads(q) for q in QUEST_PATH.read_text().splitlines() if q.strip()]
# Reads each line from quests.json1, turns it into a Python dictionary.
client = httpx.Client(timeout=60)

scores=[]
for q in qs:
    t0 = time.perf_counter()
    r = client.post(API, json={{"question": q["question"], "k":5}})
    # Send a POST request to /ask with:

    r.raise_for_status() # Wait for API response.
    ans = r.json()["answer"]
    dt = (time.perf_counter() - t0) * 1000  # in milliseconds
    hint = q.get("hint", "")
    ratio = difflib.SequenceMatcher(a=ans.lower(), b=hint.lower()).ratio()
    # Uses difflib.SequenceMatcher to get a similarity score 0.0 → 1.0 (1.0 = exact match).
    scores.append(dt)
    print(f"Q: {q['question']}\n\nLatency: {dt:.1f} ms, Score: {ratio:.2f}\n---")

log_eval_run(
    name="faiss-qwen2.5-3b",
    params={"embed_model":"all-MiniLM-L6-v2", "k": 5},
    metrics={"average_latency_ms": sum(scores)/len(scores), "total_questions": len(scores)},
)
# Logs the evaluation run to MLflow with:  

print(f"Average latency: {sum(scores)/len(scores):.1f} ms over {len(scores)} questions")

# Example Workflow
# 1. You have quests.json1 like:
# {"question":"what is this project and how do i query it?", "hint":"FastAPI, /ask endpoint"}
# {"question":"what is AI?", "hint":"Artificial intelligence"}

# 2. Your API /ask responds like:
# {"answer": "This is a FastAPI project. You can query it via /ask endpoint."}
# or
# {"answer": "Artificial intelligence is the simulation of human intelligence in machines."}

# 3. The script output might look like:
# Q: what is this project and how do i query it?
# Latency: 142.8 ms, Score: 0.85
# ---
# Q: what is AI?
# Latency: 130.5 ms, Score: 0.92
# ---
# Average latency: 136.7 ms over 2 questions