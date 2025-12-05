# 📘 **RAG Document Search (Retrieval-Augmented Generation Application)**

## 📝 **Project Summary**

RAG Document Search is an intelligent document-query system that combines vector search with LLM reasoning to deliver accurate, context-aware answers. The application ingests PDFs and URLs, converts them into vector embeddings, and stores them in a FAISS vector database for efficient semantic retrieval. When a user submits a query, the system retrieves the most relevant chunks from the vector store and passes them to the LLM for grounded response generation.

If the provided documents cannot answer the query, the application intelligently expands its search using external tools (such as Wikipedia retrieval) to supply missing context. The app also displays the exact document sources used for each answer and maintains a history of previous responses for transparency and user experience.

This project is built using Streamlit for an interactive UI and demonstrates end-to-end RAG pipeline integration for real-world document search applications.

---

## 🚀 **Features**

* **PDF & URL ingestion**
  Extracts and embeds content from user-provided documents and web pages.

* **Vector Embedding + FAISS Indexing**
  Stores embeddings in FAISS for fast semantic search and retrieval.

* **RAG-based Query Answering**
  Combines retrieved chunks with an LLM to generate accurate and grounded responses.

* **Tool-assisted External Search**
  Uses external tools such as Wikipedia when internal documents do not answer the question.

* **Source Transparency**
  Displays which document chunks were used for each answer.

* **Response History**
  Allows users to see previously generated outputs.

* **Streamlit Interactive UI**
  Simple and intuitive interface for document upload, querying, and visualization.

---

## 📂 **Project Structure**

```
DOC_RAG/
│
├── data/                 # PDF files, URLs, and raw data
├── src/                  # Core RAG pipeline & vector DB logic
│   ├── config/
│   ├── document_ingestion/
│   ├── graph_builder/
│   ├── nodes/
│   ├── states/
│   └── vectorstore/
│
├── streamlitApp.py       # Streamlit front-end application
├── requirements.txt      # Python dependencies
└── .gitignore            # Excludes env, venv, cache, etc.
```

---

## 🔧 **Tech Stack**

* **Python**
* **Streamlit** (frontend UI)
* **FAISS** (vector index)
* **OpenAI / LLM provider**
* **LangChain / custom RAG pipeline**
* **Wikipedia API tools**

---

## 🛠️ **Setup & Installation**

Clone the repository:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run streamlitApp.py
```

---

## 📌 **Usage Workflow**

1. Upload PDF or provide URLs
2. System converts documents into vector embeddings
3. FAISS index stores and retrieves relevant chunks
4. LLM generates an answer grounded in the retrieved context
5. If needed, Wikipedia is queried to fill missing knowledge
6. Sources and history are shown to the user

---

## 🎯 **Purpose of This Project**

This project serves as:

* A demonstration of end-to-end **RAG systems**
* A practical implementation of semantic document search
* A portfolio-ready **AIML engineering project**
* A foundation for chat-with-documents, enterprise knowledge bots, research assistants, and more

---
