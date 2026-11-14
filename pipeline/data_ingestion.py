import os

import boto3

from pipeline.pipeline_utils import get_logger, load_config_params

logger = get_logger(__name__)

def load_data(bucket_name: str, download_folder: str):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    file_count = 0

    try:
        logger.info(f'Loading data from S3 bucket named ${bucket_name}')

        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']

                    if key.endswith('/'):
                        continue

                    # Download file
                    file_path = os.path.join(download_folder, key)
                    s3.download_file(bucket_name, key, file_path)
                    file_count += 1

                    logger.debug(f'Downloaded file {file_path}')

        logger.debug(f"Successfully downloaded {file_count} files")

    except Exception as e:
        logger.error(f"Error listing s3 files: ${e}")
        raise

if __name__ == "__main__":
    params = load_config_params("params.yaml", logger)
    bucket_name = params['data_ingestion']['bucket_name']
    data_folder = params['utils']['data_folder']
    download_folder_path = f'{data_folder}/raw'

    os.makedirs(download_folder_path, exist_ok=True)
    load_data(bucket_name=bucket_name, download_folder=download_folder_path)
    logger.debug("Data ingestion success!")