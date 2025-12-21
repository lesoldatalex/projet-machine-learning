MLflow local helper

How to run locally (simple):

- Install dependencies: `pip install -r requirements.txt` (contains `mlflow`).
- Start MLflow server locally (SQLite backend and local artifacts dir):

  ```bash
  mlflow server \
    --backend-store-uri sqlite:///mlflow/db/mlflow.db \
    --default-artifact-root file:./mlflow/artifacts \
    --host 0.0.0.0 --port 5000
  ```

- Web UI will be available at http://localhost:5000

Notes:
- You can also use the provided `start_mlflow.sh` or `start_mlflow.bat` scripts.
- For production use consider using a proper database (Postgres) and remote artifact store.
