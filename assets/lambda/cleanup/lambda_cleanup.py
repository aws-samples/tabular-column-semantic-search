import logging

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

s3_client = boto3.client("s3")


def s3_move_objects(source_bucket, source_prefix, dest_bucket, dest_prefix):
    """Copy objects from source prefix to destination prefix and delete from source"""

    try:

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)
        source_keys = [obj["Key"] for page in page_iterator for obj in page["Contents"] if not obj["Key"].endswith(".helper")]

        for key in source_keys:

            filename = key.replace(f"{source_prefix}/", "")

            copy_response = s3_client.copy_object(CopySource={"Bucket": source_bucket, "Key": key}, Bucket=dest_bucket, Key=f"{dest_prefix}/{filename}")
            logger.debug(f"Copy response: {copy_response}")

            delete_response = s3_client.delete_object(Bucket=source_bucket, Key=key)
            logger.debug(f"Delete response: {delete_response}")

    except Exception as error:
        logger.error(f"Error: {error}")

        return error


def handler(event, context):
    """Move S3 object keys to processed locations after successful indexing"""

    try:

        bucket = event["sfn_input"]["bucket"]
        input_data_path = event["sfn_input"]["input_data_path"]
        processed_csv_s3_path = event["sfn_input"]["processed_csv_s3_path"]
        parquet_s3_path = event["sfn_input"]["parquet_s3_path"]
        embeddings_s3_path = event["sfn_input"]["embeddings_s3_path"]
        glue_output_s3URI = event["sfn_input"]["glue_output_s3URI"]
        sm_processing_output_s3URI = event["sfn_input"]["sm_processing_output_s3URI"]

        result_dict = {}

        # Move input data to processed_csv_s3_path
        res = s3_move_objects(source_bucket=bucket, source_prefix=input_data_path, dest_bucket=bucket, dest_prefix=processed_csv_s3_path)

        if res is None:
            result_dict["csv"] = {"source": input_data_path, "destination": processed_csv_s3_path}
        else:
            result_dict["csv"] = res

        # Move parquet data to parquet_s3_path
        parquet_source_prefix = glue_output_s3URI.replace(f"s3://{bucket}/", "")
        res = s3_move_objects(source_bucket=bucket, source_prefix=parquet_source_prefix, dest_bucket=bucket, dest_prefix=parquet_s3_path)

        if res is None:
            result_dict["parquet"] = {"source": parquet_source_prefix, "destination": parquet_s3_path}
        else:
            result_dict["parquet"] = res

        # Move embeddings to embeddings_s3_path
        embeddings_source_prefix = sm_processing_output_s3URI.replace(f"s3://{bucket}/", "")
        res = s3_move_objects(
            source_bucket=bucket,
            source_prefix=embeddings_source_prefix,
            dest_bucket=bucket,
            dest_prefix=embeddings_s3_path,
        )

        if res is None:
            result_dict["embeddings"] = {"source": embeddings_source_prefix, "destination": embeddings_s3_path}
        else:
            result_dict["embeddings"] = res

        logger.info("Cleanup results:")
        logger.info(result_dict)

        return result_dict

    except Exception as error:
        logger.error(f"Error: {error}")

        return error
