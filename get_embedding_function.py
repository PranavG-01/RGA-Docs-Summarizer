import os
from langchain_openai import AzureOpenAIEmbeddings

os.environ["AZURE_OPENAI_API_KEY"] = "237cb3c25a5d496287d689330c5bd2f9"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pranavopenairesource.openai.azure.com/"

def get_embedding_function():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="Pranav-Embedding-Model-4Summarizer",
        openai_api_version="2024-02-01",
    )
    return embeddings