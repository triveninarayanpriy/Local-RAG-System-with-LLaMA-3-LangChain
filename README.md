# 🧠 Local RAG System with LLaMA 3 & LangChain

> A fully offline Retrieval-Augmented Generation (RAG) pipeline that
> lets you chat with any PDF document using a locally hosted LLaMA 3
> model — no cloud, no API costs, complete data privacy.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)

## 📌 Overview

This project implements a local RAG system entirely on your machine.
It ingests a PDF, chunks and vectorises the content, stores embeddings
in ChromaDB, and answers questions using LLaMA 3 via Ollama — all
without sending data to the cloud. Built with modern LCEL.

## ✨ Key Features

- **100% Offline** — no OpenAI / Gemini API calls
- **PDF Ingestion** — PyPDFLoader for any PDF
- **Smart Chunking** — RecursiveCharacterTextSplitter (600 chars, 120 overlap)
- **Local Embeddings** — nomic-embed-text via Ollama
- **ChromaDB** — fast in-memory similarity search (top-k=3)
- **LLaMA 3 LLM** — Meta's open-source model via Ollama
- **Modern LCEL Chain** — RunnablePassthrough + StrOutputParser
- **Context-Grounded** — model answers only from retrieved chunks

## ⚙️ Prerequisites

1. Python 3.10+
2. Ollama installed — https://ollama.com
3. Pull required models:
   ollama pull llama3
   ollama pull nomic-embed-text

## 🚀 Getting Started

1. Clone the repo
   git clone https://github.com/your-username/local-rag-llama3.git

2. Install dependencies
   pip install langchain langchain-community chromadb pypdf ollama

3. Place your PDF in the project root and update PDF_FILENAME in lama1.py

4. Start Ollama: ollama serve

5. Run: python lama1.py

## 📁 Project Structure

local-rag-llama3/
├── lama1.py          # Main RAG pipeline
├── your_doc.pdf      # Your input PDF
└── README.md
