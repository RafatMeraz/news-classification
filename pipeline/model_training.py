import os
import pickle
import pandas as pd
import pathlib

from pipeline.pipeline_utils import get_logger, load_config_params

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import mlflow

logger = get_logger("model_training")

mlflow.set_experiment("news_classification")

def load_data(data_folder: str) -> pd.DataFrame:
    """
    Load data from folder and returns a combine Dataframe
    :param data_folder: Data folder path
    :return: Dataframe
    """
    try:
        dataframes = []

        folder = pathlib.Path(data_folder)
        files = [file for file in folder.iterdir() if file.is_file()]

        for file in files:
            df = pd.read_csv(file)
            dataframes.append(df)
        df = pd.concat(dataframes)
        logger.info(f"Loaded {len(dataframes)} data from {data_folder}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_folder}")
        raise
    except Exception as e:
        logger.error(e)
        raise


def save_model(model: SVC, models_folder: str, filename: str = "svc_model.pkl") -> str:
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    return model_path


def train_model(df: pd.DataFrame, target_col: str = "label") -> SVC:
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    feature_cols = [col for col in df.columns if col.startswith("tfidf_")]
    logger.info(f"Feature cols found: {len(feature_cols)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"DataFrame X shape: {X.shape}")
    logger.info(f"DataFrame Y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run() as run:
        grid_search = GridSearchCV(
            SVC(),
            param_grid,
            cv=2,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
        mlflow.log_metric("test_accuracy", report["accuracy"])

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        logger.info(f"Test accuracy: {report['accuracy']:.4f}")

        mlflow.sklearn.log_model(best_model, "svc_model")

    return best_model


def main():
    params = load_config_params("params.yaml", logger)
    data_folder = params["utils"]["data_folder"]
    encoded_data_path = f"{data_folder}/encoded"

    df = load_data(encoded_data_path)
    best_model = train_model(df)
    save_model(best_model, models_folder="models")
    logger.info("Model training done!")


if __name__ == "__main__":
    main()



