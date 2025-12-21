#!/usr/bin/env bash
set -e

# Start MLflow server with local sqlite backend and local artifacts
mkdir -p mlflow/db mlflow/artifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow/db/mlflow.db \
  --default-artifact-root file:./mlflow/artifacts \
  --host 0.0.0.0 --port 5000
