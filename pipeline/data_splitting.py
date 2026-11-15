import os
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.pipeline_utils import get_logger, load_config_params

logger = get_logger("data_splitting")


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from a single file or all CSV files in a folder.
    :param path: Path to a CSV file or a folder containing CSV files
    :return: Combined DataFrame
    """
    try:
        p = pathlib.Path(path)

        if p.is_file():
            df = pd.read_csv(p)
            logger.info(f"Loaded {len(df)} rows from {path}")
            return df

        files = [f for f in p.iterdir() if f.is_file()]
        dataframes = [pd.read_csv(f) for f in files]
        df = pd.concat(dataframes)
        logger.info(f"Loaded {len(dataframes)} files from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(e)
        raise

def split_and_save(df: pd.DataFrame, train_path: str, test_path: str, test_size: float, random_state: int):
    """
    Split combined DataFrame into train/test and save each as a single CSV.
    :param df: Full encoded DataFrame
    :param train_path: Output path for train CSV
    :param test_path: Output path for test CSV
    :param test_size: Fraction of data for test set (e.g. 0.2 = 20%)
    :param random_state: Seed for reproducibility
    """
    # stratify on label to preserve class distribution in both splits
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set: {len(train_df)} rows → {train_path}")
    logger.info(f"Test set:  {len(test_df)} rows → {test_path}")


def main():
    params = load_config_params("params.yaml", logger)
    data_folder = params["utils"]["data_folder"]
    encoded_data_path = f"{data_folder}/encoded"

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]
    train_path = f"{data_folder}/split/train.csv"
    test_path = f"{data_folder}/split/test.csv"

    df = load_data(encoded_data_path)
    split_and_save(df, train_path, test_path, test_size, random_state)
    logger.info("Data splitting done!")


if __name__ == "__main__":
    main()
