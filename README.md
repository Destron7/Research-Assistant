## 📚 RAG for Reading Research Papers

This project explores the use of **Retrieval-Augmented Generation (RAG)** to simplify and accelerate the process of reading, understanding, and summarizing research papers. It combines **local LLMs** with **semantic search** to create an intelligent assistant capable of helping with academic literature.

---

## 🔧 Prerequisites

Before running the application, make sure you have the following set up:

### ✅ 1. Install Ollama

Ollama allows you to run large language models locally.

👉 Download and install from: [https://ollama.com/download](https://ollama.com/download)

### ✅ 2. Run a Model (e.g., Deepseek-R1)

You can use Deepseek-R1 7B or another compatible model as per your requirements.

```bash
ollama run deepseek-r1:7b
```

<hr>

## 🚀 Running the Application
Once Ollama and your model are ready, follow these steps:

### 1. Install dependencies.
```bash
pip install -r requirements.txt
```

### 2. Run the main RAG Streamlit application
```bash
streamlit run rag.py
```

### 3. Alternatively, run the chatbot version
```bash
streamlit run deepseek-r1-chatbot.py
```

<hr>

## 🧪 Experiments Folder

The `Experiment/` directory contains test scripts used during initial development and trials. These include:

- 🧠 Integrations with **OpenAI APIs**
- 🗃️ Using **Pinecone** for vector storage
- 🔍 Trying different **embedding models**
- 📊 Conducting **semantic search** and **text summarization**

These files were used to explore different approaches and evaluate what worked best for this use case.

<hr>

## ✅ Conclusion

After extensive experimentation, I found that running **LLMs locally via Ollama** was the most **feasible and efficient** method for working with the RAG model.

This project helped me explore and understand:

- 🧠 **Deepseek R1** and similar models  
- 🧩 **Vector embeddings** and **similarity search**
- 📝 **Text summarization** techniques  
- ⚙️ Practical **RAG workflows**

It was a wonderful hands-on learning journey!

<hr>

## 📅 Future Plans

I plan to write a **comprehensive review paper** covering:

- 🔬 **Research on RAG and related technologies from 2020 to 2024**

Stay tuned for more academic insights!

