"""
Description:
Script to trigger Lambda function that invokes Step Functions State Machine to process batch input files

Usage:
python invoke_sfn_batch.py
"""
import argparse
import json

import boto3


def invoke(
    lambda_invoke="cdk-semantic-search-pipeline-invoke-step-functions",
    batch_prefix="data/csv/input/batch",
    region="us-east-1",
):

    lambda_client = boto3.client("lambda", region_name=region)

    lambda_input = {"batch_prefix": batch_prefix}

    try:
        response = lambda_client.invoke(
            FunctionName=lambda_invoke,
            InvocationType="RequestResponse",
            LogType="None",
            Payload=json.dumps(lambda_input),
        )

        HTTPStatusCode = response["ResponseMetadata"]["HTTPStatusCode"]

        if HTTPStatusCode == 200:
            print(f"\nSuccess! Triggered Lambda {lambda_invoke} to process batch files from {batch_prefix}\n")
        else:
            print(f"\nIssue triggering Lambda function {lambda_invoke}. Status code {HTTPStatusCode}\n")

    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_invoke", type=str, default="cdk-semantic-search-pipeline-invoke-step-functions")
    parser.add_argument("--batch_prefix", type=str, default="data/csv/input/batch")
    parser.add_argument("--region", type=str, default="us-east-1")
    args = parser.parse_args()

    lambda_invoke = args.lambda_invoke
    batch_prefix = args.batch_prefix
    region = args.region

    invoke(lambda_invoke, batch_prefix, region)
