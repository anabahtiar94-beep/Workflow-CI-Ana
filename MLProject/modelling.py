"""
Skrip pelatihan model untuk dijalankan melalui MLflow Project (CI).

Dipanggil oleh workflow CI (`mlflow run MLProject`) setiap kali trigger
terpantik, sehingga proses re-training model berjalan otomatis.
"""

import argparse

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

TARGET_COL = "target"


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="heart_disease_preprocessing/train.csv")
    parser.add_argument("--test_data", type=str, default="heart_disease_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.train_data, args.test_data)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="ci_retrain_random_forest"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))


if __name__ == "__main__":
    main()
