import os

import boto3
import numpy as np
from botocore.config import Config
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from sklearn.manifold import TSNE

region = os.environ.get("AWS_REGION")
domain = os.environ["domain_name"]

service = "es"
opensearch_client = boto3.client("opensearch", config=Config(tcp_keepalive=True))

credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, region)

# Create OS connection
res = opensearch_client.describe_domain(DomainName=domain)
host = res["DomainStatus"]["Endpoint"]

opensearch_connection = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)


def query_opensearch(connection, index, search_vector, k, payload):
    """Query OpenSearch index with knn vector"""

    query = {"size": k, "query": {"knn": {search_vector: {"vector": payload, "k": k}}}}

    res = connection.search(request_timeout=60, index=index, body=query)

    # Organize results dict
    res_keys = res["hits"]["hits"][0]["_source"].keys()
    result_dict = {}
    for key in res_keys:
        result_dict[key] = [hit["_source"][key] for hit in res["hits"]["hits"]]

    result_dict["similarity_score"] = [round(r["_score"], 4) for r in res["hits"]["hits"]]

    return result_dict


# def dimensionality_reduction
def reduce_embedding_dimensions(result_dict, search_vector, payload_embedding):
    """Reduce dimensionality of query and paylod embeddings to 2D for visualization"""

    if isinstance(payload_embedding, list):
        payload_embedding = np.asarray(payload_embedding)

    search_vector_embeddings_arr = np.array(result_dict[search_vector])
    # Add payload embedding to nn embedding array before TSNE
    search_vector_embeddings_arr_temp = np.append(search_vector_embeddings_arr, payload_embedding.reshape(-1, len(payload_embedding)), axis=0)
    # Reduce dimensionality of embeddings
    # TODO explore including dimensionality matrix of precomputed distances from opensearch knn query
    tsne_embeddings_arr = TSNE(n_components=2, verbose=0, init="pca", metric="cosine", perplexity=30, random_state=123).fit_transform(
        search_vector_embeddings_arr_temp
    )

    # Separate tsne payload and nn embeddings
    tsne_payload_list = tsne_embeddings_arr[-1].tolist()
    tsne_embeddings_arr = tsne_embeddings_arr[:-1]

    # Add reduced embeddings to result dict
    result_dict["x_reduced"] = tsne_embeddings_arr[:, 0].tolist()
    result_dict["y_reduced"] = tsne_embeddings_arr[:, 1].tolist()

    # Drop embeddings from result_dict to reduce lambda return payload size
    embedding_keys = [key for key in result_dict.keys() if "embedding" in key]
    [result_dict.pop(key, None) for key in embedding_keys]

    return tsne_payload_list, result_dict


def handler(event, context):

    index = event["index"]
    search_vector = event["search_vector"]
    k_query = event["k_query"]
    payload_embedding = event["payload_embedding"]

    lambda_return_dict = {}

    # Verify index exists
    if opensearch_connection.indices.exists(index=index):

        # Query opensearch knn with payload embedding
        result_dict = query_opensearch(opensearch_connection, index, search_vector, k_query, payload_embedding)
        # Reduce embedding dimensions
        tsne_payload_list, result_dict = reduce_embedding_dimensions(result_dict, search_vector, payload_embedding)

        # Create dict to return reduced payload and reduced query results
        lambda_return_dict["payload_reduced"] = tsne_payload_list
        lambda_return_dict["result_dict"] = result_dict
        lambda_return_dict["index_exists"] = True

    else:

        lambda_return_dict["index_exists"] = False

    # Return dict containing reduced payload and reduced query results
    return lambda_return_dict
