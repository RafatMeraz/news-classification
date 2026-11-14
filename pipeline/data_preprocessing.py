import os
import pathlib

import pandas as pd

from pipeline.pipeline_utils import get_logger, load_config_params

logger = get_logger(__name__)

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

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data by removing null rows and duplicated rows
    :param df: Dataframe
    :return: Dataframe
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def save_data(df: pd.DataFrame, file_name, data_folder: str):
    try:
        data_path = os.path.join(data_folder, file_name)
        df.to_csv(data_path)
        logger.debug(f"Saved data to {data_path}")
    except Exception as e:
        logger.error(f"Error during save data: {e}")


def process_all_files(data_folder: str, processed_data_folder: str):
    folder = pathlib.Path(data_folder)

    files = [file for file in folder.iterdir() if file.is_file()]

    for file in files:
        df = load_data(str(file.resolve()))
        df = process_data(df)
        save_data(df, file.stem, processed_data_folder)
        logger.info(f"Saved file {file.resolve()}")


def main():
    params = load_config_params("params.yaml", logger)

    data_folder = params["utils"]["data_folder"]
    raw_folder = f"{data_folder}/raw"
    processed_folder = f"{data_folder}/processed"
    os.makedirs(processed_folder, exist_ok=True)

    process_all_files(raw_folder, processed_folder)
    logger.info(f"Data pre-processing done!")


if __name__ == "__main__":
    main()
