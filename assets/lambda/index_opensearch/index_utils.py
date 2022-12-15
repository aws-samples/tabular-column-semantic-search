
import boto3
import numpy as np
import json
import io
import logging
from collections import defaultdict
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

s3client = boto3.client('s3')

def get_h5_embedding_data(file):
    embed_store = defaultdict()
    with h5py.File(file, "r") as h5:
        for key in h5.keys():
            embed_store[key] = np.array(h5.get(key))
    return embed_store

def create_index(connection, index, len_embeddings):
    '''Create index if doesn't already exist'''

    knn_index = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 256,
                "knn.space_type": "cosinesimil"
            }
        },
        "mappings": {
            "properties": {
                "column_name_embedding": {
                  "type": "knn_vector",
                  "dimension": len_embeddings,
                  "method": {
                      "name": "hnsw",
                      "engine": "nmslib",
                      "space_type": "cosinesimil",
                      "parameters": {
                          "ef_construction": 256,
                          "m": 32
                          }
                      }
                },
                "column_content_embedding": {
                  "type": "knn_vector",
                  "dimension": len_embeddings,
                  "method": {
                      "name": "hnsw",
                      "engine": "nmslib",
                      "space_type": "cosinesimil",
                      "parameters": {
                          "ef_construction": 256,
                          "m": 32
                          }
                      }
                },
                "column_name_content_embedding": {
                  "type": "knn_vector",
                  "dimension": len_embeddings * 2,
                  "method": {
                      "name": "hnsw",
                      "engine": "nmslib",
                      "space_type": "cosinesimil",
                      "parameters": {
                          "ef_construction": 256,
                          "m": 32
                          }
                      }
                }
            }
        }
    }

    if connection.indices.exists(index=index):
        logger.info("Index {index_name} already exists")
    else: 
        logger.info(f"Creating index '{index}'...")  
        res = connection.indices.create(
            index = index,
            body = knn_index,
            ignore = 400)
        logger.info(res)
    

def index_h5_embeddings_to_opensearch(connection, index, bucket, key, s3URI, table_name, embeddings_s3_path, processed_csv_s3_path):
    """
    Function to index embedding data from S3 H5 files to OpenSearch
    # # h5 file structure:
            filename (table name)
            - dataset (column name )
            -- dataset[0] (column name embedding)
            -- dataset[1] (column content embedding)
  
    Parameters:
    connection (OpenSearch connection): opensearch-py OpenSearch HTTP connection
    index (str): OpenSearch index name
    bucket (str): S3 bucket containing embedding data
    key (str): object key for h5 embedding data
    table_name (str): table name of embeddings
    embeddings_s3_path (str): path in S3 bucket where embeddings will be stored after indexing
  
    Returns:
    dict: dictionary containing num records successfully indexed
  
    """
    # Load embeddings from s3
    s3obj = s3client.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(s3obj['Body'].read())
    embed_store = get_h5_embedding_data(body)
    num_records = len(embed_store.items())
    len_embeddings = len(list(embed_store.items())[0][1][0])
    
    # Create index if doesn't already exist
    create_index(connection, index, len_embeddings)

    indexed_count = 0
    lost_count = 0
    lost_list =[]
    
    # Index csv and embedding destination keys for embeddings
    final_csv_s3URI = f's3://{bucket}/{processed_csv_s3_path}/{table_name}.csv'
    final_embeddings_s3URI = f's3://{bucket}/{embeddings_s3_path}/{table_name}.h5'

    for column_name, embeddings in embed_store.items():
        try: 
            column_name_embedding = embeddings[0]
            column_content_embedding = embeddings[1]
            column_name_content_embedding = np.append(column_name_embedding, column_content_embedding)

            connection.index(index=index,
                             body={"csv_s3URI": final_csv_s3URI,
                                   "embeddings_s3URI": final_embeddings_s3URI,
                                   "table_name": table_name,
                                   "column_name": column_name,
                                   "column_name_embedding": column_name_embedding,
                                   "column_content_embedding": column_content_embedding,
                                   "column_name_content_embedding": column_name_content_embedding})  

            indexed_count += 1

        except Exception as error:
            logger.error(f'Error for record {column_name}. Error: {error}')
            lost_count += 1
            lost_list.append(column_name)   
    
    logger.info(f'Total record count: {num_records}')
    logger.info(f'Record count indexed: {indexed_count}')
    logger.info(f'Record count not indexed: {lost_count}')
    
    if lost_count > 0:
            logger.error(f'Record(s) not indexed: {lost_list}') 
            
    result_dict = {
            's3UriIndexed': s3URI,
            'numRecords':num_records,
            'numRecordsIndexed': indexed_count,
            'numRecoredsNotIndexed': lost_count,
            'opensearchIndex': index
    }

    return result_dict
