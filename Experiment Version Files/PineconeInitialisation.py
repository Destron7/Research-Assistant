import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag"

# Fixed for OpenAI's 'text-embedding-ada-002'
# embedding_dimension = 1536

# Fixed for all-MiniLM-L6-v2 HuggingFace Model.
embedding_dimension = 384

# embedding_dimension = 512

pc.create_index(
    name=index_name,
    dimension=embedding_dimension,  # Replace with your model dimensions
    metric="cosine",    # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
