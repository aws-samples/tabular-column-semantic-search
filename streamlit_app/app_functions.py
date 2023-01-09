import json

import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def lambda_embed_payload(model_name, payload_type, payload, lambda_arn):
    """Invoke Lambda to embed user payload with SentenceTransformer model"""

    lambda_input = {"model_name": model_name, "payload_type": payload_type, "payload": payload}

    lambda_client = boto3.client("lambda")
    response = lambda_client.invoke(FunctionName=lambda_arn, InvocationType="RequestResponse", LogType="None", Payload=json.dumps(lambda_input))

    payload_embedding = json.loads(response["Payload"].read())

    return payload_embedding


def lambda_query_opensearch_and_reduce_embeddings(domain, index, search_vector, k_query, payload_embedding, lambda_arn):
    """Invoke Lambda to Query OpenSearch index with knn vector"""

    lambda_input = {
        "domain": domain,
        "index": index,
        "search_vector": search_vector,
        "k_query": k_query,
        "payload_embedding": payload_embedding,
    }

    lambda_client = boto3.client("lambda")
    response = lambda_client.invoke(FunctionName=lambda_arn, InvocationType="RequestResponse", LogType="None", Payload=json.dumps(lambda_input))

    lambda_return_dict = json.loads(response["Payload"].read())

    # Check if any knn results returned
    if lambda_return_dict["index_exists"]:

        result_dict = lambda_return_dict["result_dict"]
        payload_reduced = lambda_return_dict["payload_reduced"]

    else:

        payload_reduced = None
        result_dict = None

    return result_dict, payload_reduced


def display_results_table(result_dict, k_user_input):
    """Display results table from query results"""
    result_df = pd.DataFrame(result_dict)
    df_knn = result_df.iloc[:k_user_input][["table_name", "column_name", "similarity_score", "csv_s3URI"]]
    df_knn.columns = ["Table", "Column", "Similarity Score", "Data S3 URI"]
    # Don't show row indices - CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    # Display
    st.table(df_knn)


def display_visualization(result_dict, k_user_input, payload_reduced):
    """Dispaly 2d visualization of embeddings for query results and payload"""

    # Plot all embeddings
    result_df = pd.DataFrame(result_dict)
    hover_labels = {"table_name": True, "column_name": True, "x_reduced": False, "y_reduced": False}
    fig = px.scatter(result_df, x="x_reduced", y="y_reduced", hover_data=hover_labels, labels=dict(x_reduced="x", y_reduced="y"))
    # Plot lines to k_user_input nearest neighbors
    for i in range(k_user_input):
        column_name = result_df.iloc[i]["column_name"]
        x_nn = result_df.iloc[i]["x_reduced"]
        y_nn = result_df.iloc[i]["y_reduced"]
        fig.add_trace(go.Scatter(x=[payload_reduced[0], x_nn], y=[payload_reduced[1], y_nn], mode="lines+markers", name=column_name))
    # Plot payload
    fig.add_trace(go.Scatter(x=[payload_reduced[0]], y=[payload_reduced[1]], name="Payload", legendrank=1))
    # Display
    fig.update_layout(title_text="EDA Visualization*", title_x=0.5)
    # fig.update_xaxes(visible=False)
    st.plotly_chart(fig)
    st.markdown(
        "<p style='text-align: left; color: grey;'>*2D visualization may skew perceived \
            nearest neighbors due to dimensionality reduction.</p>",
        unsafe_allow_html=True,
    )
