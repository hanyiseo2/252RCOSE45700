# AWS Well Architected Framework Chatbot

## 📌 Overview

This project is a RAG-based chatbot built using LangChain and FAISS.
It loads three AWS Well-Architected PDF documents (~800 pages total), generates vector embeddings, and retrieves context-relevant chunks to answer user questions with source references.
The UI is implemented with Gradio.

## 🚀 Features

* RAG (Retrieval-Augmented Generation) architecture
* Embedding generation using OpenAIEmbeddings
* Vector store using FAISS
* Source citation for each answer
* Multi-document ingestion (3 PDFs > 600 pages)
* Gradio-based interactive chat UI
* Python 3.12 + virtual environment

## 🛠 Tech Stack

| Category | Tools                 |
| -------- | --------------------- |
| Backend  | FastAPI, Python 3.12  |
| AI/ML    | LangChain, OpenAI API |
| Infra    | AWS EC2, IAM          |
| etc      | GitHub, Gradio        |

---

## ⚙️ Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd 252RCOSE45700

# 2. Create Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file and add:
# OPENAI_API_KEY=xxxxxx

# 5. Ingest documents (build FAISS vector store)
python3 ingestion.py

# 6. Run Chatbot UI
python3 chatbot_ui.py
```
