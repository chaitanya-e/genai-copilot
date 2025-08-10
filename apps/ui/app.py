import streamlit as st, requests, os, time
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="GenAI Copilot", page_icon=":robot_face:", layout="centered")
st.title("GenAI Copilot")

if "history" not in st.session_state:
    st.session_state.history = []
    # Uses Streamlit’s session state to store all previous questions and answers in a list called history.
    # This way, history persists across UI reruns while you interact.

q=st.text_input("Ask a question about your docs")
if st.button("Ask") and q.strip():
    t0 = time.time()
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": q, "k": 5}, timeout = 60)
        if r.status_code == 200:
            data = r.json()
            st.session_state.history.append((q,data))
        else:
            st.error(f"Error: {r.status_code} - {r.text}")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
# When the user clicks the Ask button and the question is not empty:
# Records the start time t0 (you measure latency but currently don’t use it explicitly here).
# Sends a POST request to your backend /ask endpoint with:
# The question text (q)
# k=5 (maybe top 5 results or context size)
# Waits up to 60 seconds for a response.
# If success (200), it:
# Parses JSON response (data)
# Appends (question, data) tuple to history.
# If failure, shows an error message.
# Catches request exceptions (e.g., timeout, network issues).

for q,data in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer:** {data['answer']}")
    with st.expander("Citations"):
        for s in data["sources"]:
            path= s.get("path") or "N/A"
            st.write(f"[{s.get('id', 'N/A')}] {s.get('title', 'No Title')} - {path}")
    st.caption(f"Latency: {data['latency_ms']} ms")
    st.markdown("---")

# Loops backwards over the history (most recent first).
# For each question-answer pair:
# Shows your question (You:)
# Shows AI’s answer (Answer:)
# Shows an expandable section with citations:
# For each citation source, prints its ID, title, and file path.
# Shows latency (how long the backend took to answer).
# Separates entries with a horizontal line (---).