import pickle

import pandas as pd
import mlflow

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from pipeline.pipeline_utils import get_logger, load_config_params, init_mlflow

logger = get_logger("model_evaluation")

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the data from file
    :param data_path: path to file
    :return: Dataframe
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error during read csv: {e}")
        raise

def load_model(model_path: str) -> SVC:
    """
    Load a pickled SVC model from disk.
    :param model_path: Path to .pkl file
    :return: Deserialized SVC model
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def evaluate_model(
    model: SVC,
    df: pd.DataFrame,
    run_id: str,
    target_col: str = "label",
) -> dict:
    """
    Evaluate model on full dataset and log metrics to an existing MLflow run.
    :param model: Trained SVC model
    :param df: Encoded DataFrame with tfidf_ feature columns and target column
    :param run_id: MLflow run_id from training to log metrics into
    :param target_col: Name of the target label column
    :return: Dict of evaluation metrics
    """
    feature_cols = [col for col in df.columns if col.startswith("tfidf_")]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    X = df[feature_cols]
    y = df[target_col]

    y_pred = model.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    # macro f1 treats all classes equally regardless of support size
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    metrics = {
        "eval_accuracy": accuracy,
        "eval_f1_macro": f1_macro,
        "eval_f1_weighted": f1_weighted,
    }

    # Per-class precision, recall, f1
    for label, values in report.items():
        if isinstance(values, dict):
            metrics[f"eval_precision_{label}"] = values["precision"]
            metrics[f"eval_recall_{label}"] = values["recall"]
            metrics[f"eval_f1_{label}"] = values["f1-score"]

    # Reopen training run to keep all metrics in one place
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        logger.info(f"Evaluation metrics logged to run {run_id}")

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 macro: {f1_macro:.4f} | F1 weighted: {f1_weighted:.4f}")
    logger.info(f"Confusion matrix:\n{cm}")

    return metrics

def load_run_id() -> str:
    with open("models/run_id.txt", "r") as f:
        run_id = f.read().strip()
    logger.info(f"Loaded run_id: {run_id}")
    return run_id


def main():
    params = load_config_params("params.yaml", logger)
    data_folder = params["utils"]["data_folder"]
    test_path = f"{data_folder}/split/test.csv"
    model_path = params.get("model_evaluation", {}).get("model_path", "models/svc_model.pkl")

    run_id = load_run_id()

    init_mlflow(params, logger)

    df = load_data(test_path)
    model = load_model(model_path)
    evaluate_model(model, df, run_id=run_id)
    logger.info("Model evaluation done!")


if __name__ == "__main__":
    main()
