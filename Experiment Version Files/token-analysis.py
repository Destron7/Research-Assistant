import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

# Load the PDF
loader = PyPDFLoader(r"data/impact_of_generativeAI.pdf")
document = loader.load()

# Combine text from all pages
text = "\n".join([doc.page_content for doc in document])

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_text(text)

print(texts)

# Count tokens for each chunk and display them
total_tokens = 0

print("\nToken count per chunk:\n")
for i, chunk in enumerate(texts):
    chunk_tokens = len(tokenizer.encode(chunk))
    total_tokens += chunk_tokens
    print(f"Chunk {i + 1}: {chunk_tokens} tokens")

# Display total token count
print(f"\nTotal tokens in document: {total_tokens}")
