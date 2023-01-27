import json
import os
from datetime import datetime

import boto3

client = boto3.client("stepfunctions")

# Get environment variables
account_id = os.environ["account_id"]
sfn_arn = os.environ["state_machine_arn"]
resources_name_prefix = os.environ["resources_name_prefix"]
parquet_s3_path = os.environ["parquet_s3_path"]
processed_csv_s3_path = os.environ["processed_csv_s3_path"]
embeddings_s3_path = os.environ["embeddings_s3_path"]
bucket = os.environ["bucket"]


def handler(event, context):

    if "batch_prefix" in event:
        input_type = "batch"
        key = event["batch_prefix"]
        input_data_path = key
    else:
        input_type = "file"
        key = event["Records"][0]["s3"]["object"]["key"]
        input_data_path = "data/csv/input/file"

    job_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    input = {
        "sfn_input": {
            "input_type": input_type,
            "input_s3URI": f"s3://{bucket}/{key}",
            "bucket": bucket,
            "input_data_path": input_data_path,
            "parquet_s3_path": parquet_s3_path,
            "processed_csv_s3_path": processed_csv_s3_path,
            "embeddings_s3_path": embeddings_s3_path,
            "glue_output_s3URI": f"s3://{bucket}/{parquet_s3_path}/tmp/{job_timestamp}",
            "sm_processing_jobname": job_timestamp,
            "sm_processing_output_s3URI": f"s3://{bucket}/{embeddings_s3_path}/tmp/{job_timestamp}",
        }
    }

    # Start Step Functions State Machine
    client.start_execution(stateMachineArn=sfn_arn, input=json.dumps(input))
