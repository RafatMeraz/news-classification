import os
import pickle

import pandas as pd

from pipeline.pipeline_utils import get_logger, load_config_params, init_mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import mlflow

logger = get_logger("model_training")


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

def save_model(model: SVC, models_folder: str, filename: str = "svc_model.pkl") -> str:
    """
    Serialize model to disk as a pickle file.
    :param model: Trained SVC model
    :param models_folder: Destination folder (created if missing)
    :param filename: Output filename
    :return: Full path to saved model file
    """
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    return model_path


def train_model(df: pd.DataFrame, params: dict, target_col: str = "label") -> tuple[SVC, str]:
    """
    Train SVC using GridSearchCV hyperparameter tuning and log results to MLflow.
    Expects pre-split train data — do not pass full dataset.
    :param df: Train-split DataFrame with tfidf_ feature columns and target column
    :param params: Config params containing model_train settings (cv, scoring, n_jobs)
    :param target_col: Name of the target label column
    :return: Tuple of (best SVC estimator, MLflow run_id)
    """
    logger.info(f"DataFrame shape: {df.shape}")

    # All tfidf_ columns produced by feature engineering are used as features
    feature_cols = [col for col in df.columns if col.startswith("tfidf_")]
    logger.info(f"Feature cols found: {len(feature_cols)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"X shape: {X.shape} | y shape: {y.shape}")

    param_grid = {
        "C": [0.1, 1, 10, 100],       # regularization strength
        "kernel": ["linear", "rbf"],   # linear for text features; rbf for non-linear boundaries
        "gamma": ["scale", "auto"],    # only applies to rbf kernel
    }

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, log_models=False)

    with mlflow.start_run() as run:
        grid_search = GridSearchCV(
            SVC(),
            param_grid,
            cv=params['model_train']['cv'],
            scoring=params['model_train']['scoring'],
            n_jobs=params['model_train']['n_jobs'],
            verbose=1,
        )
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        mlflow.sklearn.log_model(best_model, "svc_model")

        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")

        run_id = run.info.run_id


    return best_model, run_id

def save_run_id(run_id: str):
    try:
        with open("models/run_id.txt", "w") as f:
            f.write(run_id)
        logger.debug(f"Run ID saved: {run_id}")
    except Exception as e:
        logger.error(f"Error during save run_id: {e}")
        raise

def main():
    params = load_config_params("params.yaml", logger)
    data_folder = params["utils"]["data_folder"]
    train_path = f"{data_folder}/split/train.csv"

    init_mlflow(params, logger)

    df = load_data(train_path)
    best_model, run_id = train_model(df, params=params)
    save_model(best_model, models_folder="models")

    save_run_id(run_id)

    logger.info(f"MLflow run_id saved: {run_id}")
    logger.info("Model training done!")


if __name__ == "__main__":
    main()



