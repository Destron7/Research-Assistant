import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }

    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }

    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }

    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }

    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant.
Use the Provided to answer the question.
If the answer is not in the provided text, say "I don't know".
Be concise & Factual.
try giving answer in points and more concise way.

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_PATH = 'data/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:14b")
OLLAMALLM = OllamaLLM(model="deepseek-r1:14b")

# documentation link for inmemory vector store:
# https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html
VECTOR_STORE = InMemoryVectorStore(EMBEDDING_MODEL)

def save_uploaded_file(uploaded_file):
    with open(PDF_PATH + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return PDF_PATH + uploaded_file.name

def load_pdf(file_path):
    document_loader = PDFPlumberLoader(file_path)
    document = document_loader.load()
    return document

# Document Data Chunking.
def chunk_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(document)

def index_documents(document_chunks):
    VECTOR_STORE.add_documents(document_chunks)

def find_similar_documents(query):
    similar_docs = VECTOR_STORE.similarity_search(query, k=5)
    return similar_docs

def generate_answer(query, context_docs):
    context_text = "\n".join([doc.page_content for doc in context_docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = prompt | OLLAMALLM
    return response_chain.invoke({"user_query": query, "document_context": context_text})

st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf(saved_path)
    processed_chunks = chunk_document(raw_docs)
    index_documents(processed_chunks)

    st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = find_similar_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
