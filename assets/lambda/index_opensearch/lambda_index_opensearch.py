import boto3
from botocore.config import Config
from requests_aws4auth import AWS4Auth
import os
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from pathlib import Path
from index_utils import index_h5_embeddings_to_opensearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get environment variables
region = os.environ.get('AWS_REGION')
domain_name = os.environ['domain_name']
processed_csv_s3_path = os.environ['processed_csv_s3_path']

opensearch_client = boto3.client('opensearch', config=Config(tcp_keepalive=True))
s3client = boto3.client('s3')

service = 'es'
credentials = boto3.Session().get_credentials()
auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

res = opensearch_client.describe_domain(DomainName=domain_name)
host = res['DomainStatus']['Endpoint']

# Create OS connection
opensearch_connection = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

def handler(event, context):
    '''Index table column embeddings from S3 to OpenSearch'''

    bucket = event['sfn_input']['bucket']
    embeddings_s3_path = event['sfn_input']['embeddings_s3_path']
    sm_processing_output_s3URI = event['sfn_input']['sm_processing_output_s3URI']
    processing_prefix = sm_processing_output_s3URI.replace(f's3://{bucket}/','')

    # Get embeddings key
    objects = s3client.list_objects_v2(Bucket=bucket, Prefix=processing_prefix)['Contents']
    key_list = [obj['Key'] for obj in objects]
    
    result_dict = {}
    
    for key in key_list:
    
        embeddings_s3URI = f's3://{bucket}/{key}'
        logger.info(f'Indexing embeddings from {key}')

        table_name = Path(key).stem
    
        # Set OpenSearch index name
        index = key \
            .replace(f'{processing_prefix}/', '') \
            .replace(f'/{table_name}.h5','') \
            .replace('/','-') \
            .replace('_','-') \
            .lower() 
    
        if index not in result_dict:
            result_dict[index] = {}

        result_dict[index][table_name] = index_h5_embeddings_to_opensearch(opensearch_connection, index, bucket, key, embeddings_s3URI, table_name, embeddings_s3_path, processed_csv_s3_path)

    return result_dict