@echo off
REM Start MLflow server (Windows)
if not exist mlflow\db mkdir mlflow\db
if not exist mlflow\artifacts mkdir mlflow\artifacts
mlflow server --backend-store-uri sqlite:///mlflow/db/mlflow.db --default-artifact-root file:./mlflow/artifacts --host 0.0.0.0 --port 5000
