"""
Description:
Script to invoke Lambda function that invokes Step Functions State Machine to process batch input files

Usage:
Batch upload and invoke Lambda (default) 
        python run_pipeline.py --destination_bucket <DESTINATION_BUCKET>
        
Single file upload. 
"""
import argparse
import json
import pandas as pd
import boto3
from utils import configure_logging
from pathlib import Path
import logging

_ = configure_logging("run-pipeline")

def invoke(batch_prefix:str="data/csv/input/batch", region:str="us-east-1"):

    lambda_client = boto3.client("lambda", region_name=region)
    lambda_input = {"batch_prefix": batch_prefix}
    lambda_fn_name = [f for f in lambda_client.list_functions()["Functions"] if "invoke-step-functions" in f["FunctionName"]][0]["FunctionName"]

    try:
        response = lambda_client.invoke(
            FunctionName=lambda_fn_name,
            InvocationType="RequestResponse",
            LogType="None",
            Payload=json.dumps(lambda_input),
        )

        HTTPStatusCode = response["ResponseMetadata"]["HTTPStatusCode"]

        if HTTPStatusCode == 200:
            logging.info(f"\nSuccess! Triggered Lambda {lambda_fn_name} to process batch files from {batch_prefix}\n")
        else:
            logging.info(f"\nIssue triggering Lambda function {lambda_fn_name}. Status code {HTTPStatusCode}\n")

    except Exception as e:
        logging.error(e)

def _run_batch_pipeline(batch_datasets_file:str, destination_bucket:str, destination_prefix:str, region:str, max_rows:int):
    
    with Path(batch_datasets_file).open("r") as f:
        sample_datasets = json.load(f)
        
    for name, url in sample_datasets.items():
        df = pd.read_csv(url, nrows=max_rows)
        fn = f"s3://{destination_bucket}/{destination_prefix}/{name}.csv"
        df.to_csv(fn, index=False)
        logging.info(f"Uploaded {name} to {fn}")

    invoke(destination_prefix, region)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload CSV file(s) to S3 and invoke pipeline.")
    parser.add_argument("--destination_bucket", type=str, required=True)
    parser.add_argument("--destination_prefix", type=str, default="data/csv/input")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS service region.")
    parser.add_argument("--input_mode", type=str, choices=["file", "batch"], default="batch")
    parser.add_argument("--batch_datasets_file","-b", type=str, default="sample-batch-datasets.json", help="JSON file containing URLs of sample datasets.")
    parser.add_argument("--file_or_url", "-f", type=str, default=None, help="Local path or remote URL to CSV file to upload.")
    
    parser.add_argument("--max_rows", default=2**14, type=int, help="Maximum number of rows to download.")

    args = parser.parse_args()
    destination_prefix = f"{args.destination_prefix}/{args.input_mode}"
    destination_bucket = args.destination_bucket
    region = args.region
    max_rows = args.max_rows
    
    if args.input_mode == "batch":
        logging.info("Batch CSV mode.")
        _run_batch_pipeline(args.batch_datasets_file, destination_bucket, destination_prefix, region, max_rows)
    
    if args.input_mode == "file" and args.file_or_url:
        logging.info("Single CSV mode.")
        df = pd.read_csv(args.file_or_url, nrows=max_rows)
        fn = f"s3://{destination_bucket}/{destination_prefix}/{Path(args.file_or_url).stem}"
        df.to_csv(fn, index=False)
        logging.info(f"Uploaded {args.file_or_url} to {fn}")
    
    

