import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Loading out preferred Pdf Document file to train our pre-trained model.
# for loading pdfs PyPDFLoader is used.
"""
File Type	            Recommended Loader
Text	                TextLoader
PDF	                    PyPDFLoader
Word (.docx)	        UnstructuredWordDocumentLoader
Excel (.xlsx)	        UnstructuredExcelLoader
CSV	                    CSVLoader
Webpages	            WebBaseLoader
YouTube Transcripts	    YoutubeLoader
Google Drive	        GoogleDriveLoader
JSON	                JSONLoader

Other supported Loadable File Types.
"""

# Change path of the file here which you want to feed to the model.

# loader = PyPDFLoader(r"data/impact_of_generativeAI.pdf")
loader = PDFPlumberLoader(r"data/impact_of_generativeAI.pdf")
document = loader.load()

# Breaking down the document into smalled chunks.
"""
Why Split Document?
    -Preserving Context of the Flow.
    -Processing Data in Memory Efficient Way.
    -Respecting Token Limits of the Pre-trained model.
"""

"""
Understanding Chunk Size & Chunk Overlap is really important for working with a RAG Model.

Chunk Size: 
    Number of maximum characters in each Chunk.
Chunk Overlap: 
    Characters Allowed to overlap between consecutive Chunks to maintain Lingual context among them.
    
VIDEO SOURCE: https://youtu.be/LuhBgmwQeqw
NOTE:   I found the resource helpful it can be different for you. Please tally & recheck the information 
        with the official langchain or Pinecone documentation.    
"""

text = "\n".join([doc.page_content for doc in document])

"""
CharacterTextSplitter needs lines & paragraph breakpoints to chunk precisely.
instead use RecursiveCharacterTextSplitter to make the proces precise & accurate.
"""
# text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(text)

# texts = texts[:5]

print(f"created Chunks : {len(texts)}")

# Creating Vector Embeddings

"""
Using OpenAI’s language models to create vector representations (embeddings) that capture their meaning.

    OpenAIEmbeddings: LangChain Class used for interacting with OpenAI's language models.
    
    PineconeVectorStore: This Class connects LangChain with pinecone vector database.
"""

# If you want to use OpenAI text-embedding-3-large for generation of Vector Embeddings.
"""
embeddings = OpenAIEmbeddings(
    # openai_api_type=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-large"
)
"""

# Using HuggingFace all-MiniLM-L6-v2 model for Creating Vector Embeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Using all-MiniLM-L6-v2 by default
chunk_embeddings = embeddings.embed_documents(texts)


# PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
PineconeVectorStore.from_texts(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))