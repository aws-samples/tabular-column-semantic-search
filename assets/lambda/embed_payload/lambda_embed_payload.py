import numpy as np
from sentence_transformers import SentenceTransformer

def handler(event, context):
    '''Embed user payload with SentenceTransformer model'''
    
    model_name = event['model_name']
    payload_type = event['payload_type']
    payload = event['payload']

    # Get SentenceTransformer embedding model
    embedding_model = SentenceTransformer(model_name, cache_folder='/tmp/cache/')

    # Create concatenated embedding payload if payload_type == "Column Name: Column Content"
    if payload_type == "Column Name: Column Content":
        # Split payload on colon and strip white space from beginning/end
        splitted_payload = [item.strip() for item in payload.split(':')]
        col_name = splitted_payload[0]
        col_content = splitted_payload[1]
        # Embed column name and content separately
        name_embedding = embedding_model.encode(col_name)
        content_embedding = embedding_model.encode(col_content)
        # Set payload_embedding as concatenated name and column embeddings
        payload_embedding = np.append(name_embedding, content_embedding)
    else: 
        payload_embedding = embedding_model.encode(payload)
        
    return payload_embedding.tolist()
    