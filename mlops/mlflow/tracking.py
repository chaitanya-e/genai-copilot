import os, mlflow, time

def log_eval_run(name: str, params: dict, metrics: dict):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    with mlflow.start_run(run_name=name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_text(time.strftime("%Y-%m-%d %H:%M:%S"), "timestamp.txt")