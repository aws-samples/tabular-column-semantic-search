import sys
import boto3
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")
# logger = logging.getLogger("glue-convert-parquet-convert")
logger = logging.getLogger()

logger.setLevel(logging.INFO)

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME','input_s3URI','glue_output_s3URI'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

input_S3_URI = args['input_s3URI']
output_S3_URI = args['glue_output_s3URI']

if input_S3_URI.endswith('.csv'):
    s3_uris = [input_S3_URI]
else:
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator('list_objects_v2')
    bucket = input_S3_URI.split("s3://")[1].split("/")[0]
    prefix = input_S3_URI.replace(f"s3://{bucket}/",'')
    # Get keys from objects at input_S3_URI
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    s3_uris = [f"s3://{bucket}/{obj['Key']}" for page in page_iterator for obj in page["Contents"]]  

for input_S3_URI in s3_uris:
    if not ".csv" in input_S3_URI:
        continue
    
    logging.info(f"Converting {input_S3_URI} to Parquet")
    table_name = Path(input_S3_URI).stem
    
    input_df = glueContext.create_dynamic_frame_from_options(
        connection_type = "s3", 
        connection_options = {
            "paths": [input_S3_URI]
        }, 
        format = "csv",
        format_options={
            "withHeader": True,
            "separator": ","
        }
    )

    destination_s3_uri = f"{output_S3_URI}/{table_name}"

    output_df = glueContext.write_dynamic_frame.from_options(
        frame = input_df,
        connection_type = "s3",
        connection_options = {
            "path": destination_s3_uri,
            "partitionKeys": []
        }, 
        format = "parquet",
        format_options={"compression": "snappy"}
    )

    logging.info(f"Converted {input_S3_URI} to Parquet: {destination_s3_uri}")
      
job.commit()