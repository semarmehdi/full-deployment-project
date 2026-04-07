import argparse
import time
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


load_dotenv()

# Tracking server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    experiment_name = "iris_classifier"
    registered_model_name = "iris_classifier"
    alias_name = "challenger"

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    print("Training model...")
    start_time = time.time()

    # Keep autolog basic for a deployment/demo course, but log the model manually
    mlflow.sklearn.autolog(log_models=False)

    # ------------------------------------------------------------------
    # Dataset: Iris
    # ------------------------------------------------------------------
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Model pipeline
    # ------------------------------------------------------------------
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    min_samples_split=args.min_samples_split,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_param("dataset", "iris")

        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)

        # MLflow 3.x: prefer `name=` instead of deprecated `artifact_path=`
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        model_version = model_info.registered_model_version
        print(f"[INFO] Model logged as version {model_version}")

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=model_version,
        )
        print(
            f"[INFO] Alias '{alias_name}' now points to version {model_version}"
        )

        # Optional: handy tags for the registry/UI
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="dataset",
            value="iris",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="metric:test_accuracy",
            value=f"{accuracy:.4f}",
        )

        print(f"[INFO] Run ID: {run.info.run_id}")
        print(f"[INFO] Test accuracy: {accuracy:.4f}")

    print("...Done!")
    print(f"--- Total training time: {time.time() - start_time:.2f} seconds")
