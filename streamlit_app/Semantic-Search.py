import streamlit as st
import streamlit_authenticator as stauth
import yaml
import os
from app_functions import *

####################
# Set opensearch domain name and lambda function ARNs
domain_name = os.environ['opensearch_domain']
lambda_embed_arn = os.environ['lambda_embed_arn']
lambda_query_arn = os.environ['lambda_query_arn']

# Set k nearest neighbors to return from opensearch, used for result visualization
k_query = 500 
####################
st.set_page_config(layout="wide")

# Display title
# st.title("Tabular Column Semantic Search")
st.markdown("<h1 style='text-align: center;'>Tabular Column Semantic Search</h1>", unsafe_allow_html=True)
st.markdown("""---""")

col1, col2 = st.columns((1, 3), gap='large')

def main():
    '''App runner'''

    ####################
    # User inputs
    with col1:

        st.header("User Inputs")

        payload_type = st.radio("Payload type", 
                                ("Column Name", "Column Content", "Column Name: Column Content"),
                                horizontal=False)

        # Set default_payload and search_vector based on payload_type
        if payload_type == "Column Name":
            default_payload = "address"
            search_vector = 'column_name_embedding' 
        elif payload_type == "Column Content":
            default_payload = "5412 Erin Valleys, Shieldsmouth, KS 71187"
            search_vector = 'column_content_embedding'
        elif payload_type == "Column Name: Column Content":
            default_payload = "address: 5412 Erin Valleys, Shieldsmouth, KS 71187"
            search_vector = 'column_name_content_embedding'

        payload = st.text_input("Payload", default_payload)

        model_name = st.radio("Embedding model", 
                            ("all-MiniLM-L6-v2", 
                            "all-distilroberta-v1", 
                            "average_word_embeddings_glove.6B.300d"))
                            
        k_user_input = st.slider("Value of K", min_value=1, max_value=30, value=15)

    ####################
    # Output
    with col2:
        # st.markdown("""---""")

        # st.header("Search Results")
        st.markdown("<h2 style='text-align: center;'>Search Results</h2>", unsafe_allow_html=True)

        if not payload.strip():
            st.caption("Please provide payload")
        elif payload_type == "Column Name: Column Content" and ':' not in payload:
            st.caption("Please input Column Name and Column Content separated by a colon, e.g. '<Column Name>: <Column Content>' ")
        else:
            with st.spinner('Searching...'):

                # Query opensearch index with embedded payload
                index_name = f"{model_name}-batch-nrows-1024".replace('_','-').lower()  

                # payload_embedding = embed_payload(model_name, payload_type, payload)
                payload_embedding = lambda_embed_payload(model_name, payload_type, payload, lambda_embed_arn)
                result_dict, payload_reduced = lambda_query_opensearch_and_reduce_embeddings(domain_name, index_name, search_vector, k_query, payload_embedding, lambda_query_arn)
                
                # Return nearest neighbors if they exist, ie embeddings have been indexed to OpenSearch
                if result_dict is not None:

                    result_count = len(result_dict['column_name'])

                    if k_user_input > result_count:
                        st.warning(f"K={k_user_input} while OpenSearch contains only {result_count} column embeddings for the selected model.")
                        k_user_input = result_count
                        
                    st.success(f"{k_user_input} nearest neighbors for {payload_type} = '{payload}', using embeddings from {model_name}:")   
                    display_results_table(result_dict, k_user_input)
                    # display_visualization(result_dict, k_user_input, payload_reduced)
                    if 'party' in payload: st.balloons()
                
                else:
                    st.warning("No results found. Make sure OpenSearch contains indexed embeddings for the selected model.")

def auth_main(): 
    '''Authenticate user and run app'''

    with open('auth.yaml') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        main()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    auth_main()