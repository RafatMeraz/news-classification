import json
import os

import boto3

from pipeline.pipeline_utils import get_logger, load_config_params

logger = get_logger("s3 manifest")


def write_s3_manifest(bucket_name: str, output_path: str):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    objects = []

    try:
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if not obj['Key'].endswith('/'):
                        objects.append({
                            'key': obj['Key'],
                            'etag': obj['ETag'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                        })

        objects.sort(key=lambda x: x['key'])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(objects, f, indent=2)

        logger.debug(f"Manifest written with {len(objects)} objects")

    except Exception as e:
        logger.error(f"Error fetching S3 manifest: {e}")
        raise


if __name__ == "__main__":
    params = load_config_params("params.yaml", logger)
    bucket_name = params['data_ingestion']['bucket_name']
    write_s3_manifest(bucket_name=bucket_name, output_path='data/s3_manifest.json')