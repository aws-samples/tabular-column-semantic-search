# Tabular Column Semantic Search

#### App receives user payload, queries OpenSearch for KNN embeddings, and visualizes results
#### Steps to run locally
1. Create virtual environment
```
    python -m venv .venv
```
2. Activate virtual environment
```
    source .venv/bin/activate
```
3. Install libraries in requirements.txt
```
    pip install -r requirements.txt
```
4. Set variables
```
    Manually set the following variables in Semantic-Search.py
    - domain_name
    - lambda_embed_arn
    - lambda_query_arn
```
5. Run app
```
    streamlit run app.py
```
