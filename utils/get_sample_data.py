"""
Description:
Script to copy sample data to specified bucket

Usage:
python get_sample_data.py --destination_bucket=<DESTINATION_BUCKET> --input_type=<file | batch>

"""

import argparse

import invoke_sfn_batch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--destination_bucket", type=str, required=True)
parser.add_argument("--input_type", type=str, choices=["file", "batch"], default="file")
args = parser.parse_args()

source_datasets = {
    "chicago_speed_camera_violations.csv": "https://data.cityofchicago.org/resource/hhkd-xvj4.csv",
    "nypd_complaint_data_historic.csv": "https://data.cityofnewyork.us/resource/qgea-i56i.csv",
    "nyc_311_service_requests_2009.csv": "https://data.cityofnewyork.us/resource/3rfa-3xsf.csv",
}

# Set num records to return from datasets
record_limit = 100000

destination_bucket = args.destination_bucket
input_type = args.input_type

if input_type == "file":
    destination_prefix = "data/csv/input/file"
    num_datasets = 1
elif input_type == "batch":
    destination_prefix = "data/csv/input/batch"
    num_datasets = len(source_datasets)

print("\nUploading data...")

i = 0
for key in source_datasets.keys():
    i += 1
    if i > num_datasets:
        continue

    # Load dataframe
    url = f"{source_datasets[key]}?$limit={record_limit}"
    df = pd.read_csv(url)

    # Upload to S3
    s3URI = f"s3://{destination_bucket}/{destination_prefix}/{key}"
    df.to_csv(s3URI, index=False)

    print(f"\nUploaded {url} to {s3URI}")

print("\nData upload complete")

if input_type == "batch":
    invoke_sfn_batch.invoke()
