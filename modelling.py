import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.csv")
    parser.add_argument("--test-data", default="data/test.csv")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    target = "diagnosis"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    model = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": args.c,
            "max_iter": args.max_iter,
            "random_state": args.random_state,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": X_train.shape[1],
        })
        mlflow.log_metrics(metrics)

        project_root = Path(__file__).resolve().parent
        output_dir = project_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "run_id.txt").write_text(run_id)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        report_path = output_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print("Evaluation metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
