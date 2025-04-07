from pinecone import Pinecone
import os
from dotenv import load_dotenv
import ollama

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Initialize Pinecone
# pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create index (if not exists)
index_name = os.environ.get("INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=768, metric="cosine")
else:
    print(f"Vector Database Index Found: {index_name}")

# Connect to index
index = pc.Index(index_name)

# Query a similar document
query_text = "What are the types of Generative Models!?"

# For Retrieving Vector Embeddings from Pinecone Vector Storage the model which should be used must be
# same as the model which was used for generating embeddings to match the dimensions of the embeddings.

hFEmbeddingModel = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Using all-MiniLM-L6-v2 by default
query_embeddings = hFEmbeddingModel.embed_documents([query_text])

# query_embeddings_output = ollama.embeddings(
#     prompt=query_text,
#     model='nomic-embed-text'
# )

query_embedding = query_embeddings[0]

# Search in Pinecone
results = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)

# Preparing the prompt for summarization
matchsummary_text = "\n".join(match["metadata"]["text"] for match in results["matches"])
matchsummary_text = f'Summarize the following text concisely:\n {matchsummary_text}\n I don\'t need any extra text from your side give me important content only.\nDon\'t give me <think></think> part in response'

summary_response = ollama.chat(
    model="deepseek-r1:14b",
    messages=[{"role": "user", "content": matchsummary_text}]
)

summarized_response_string = summary_response.message["content"].split("</think>")[1].strip()
print(summarized_response_string)
