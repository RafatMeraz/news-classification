import logging
import os
from logging import Logger

import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get the Logger object for logging
    :param name: The name of the logger
    :param level: The logging level
    :return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    log_file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def load_config_params(path: str, logger: Logger) -> dict:
    """Loads the configuration params from YAML file"""
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Params has been retrieved")
        return params
    except FileNotFoundError:
        logger.error("Config file not found")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"YAML error: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise

