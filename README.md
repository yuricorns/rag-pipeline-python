# RAG Pipeline — Document Q&A in Pure Python

A Retrieval Augmented Generation (RAG) system built from scratch in pure Python.
Ask any question about a document and get accurate answers powered by Groq LLM.
Built without LangChain or any high-level AI frameworks — every component written manually.

## What is RAG?
RAG combines the search power of vector databases with the language power of LLMs.
Instead of relying on the LLM's training knowledge, RAG retrieves relevant information
from your own documents and uses that as context to generate accurate answers.

## How it works
1. Document is split into chunks (sentences/paragraphs)
2. Each chunk is converted into embedding numbers and stored in ChromaDB
3. User asks a question
4. Question is converted to embeddings and matched against stored chunks
5. Most relevant chunks are retrieved
6. Chunks + question are sent to Groq LLM as context
7. LLM generates an accurate answer based only on the document

## Tech Stack
- Python
- ChromaDB — vector database for storing and searching embeddings
- Groq LLM (llama-3.3-70b-versatile) — for generating answers
- Sentence similarity search — semantic matching not keyword matching

## Features
- Semantic search — finds relevant content by meaning not exact keywords
- Grounded answers — LLM only answers from document context
- Hallucination prevention — returns "I don't know" if answer not in document
- Pure Python implementation — no LangChain or automation tools

## How to Run

1. Clone the repo
   git clone https://github.com/yuricorns/rag-pipeline-python

2. Install dependencies
   pip install chromadb groq

3. Add your Groq API key in rag.py

4. Add your content to knowledge.txt

5. Run
   python rag.py

6. Ask any question about the content in knowledge.txt

## Example
Question: What is ChromaDB used for?
Answer: ChromaDB is a vector database used to store embeddings.

Question: What is RAG?
Answer: RAG stands for Retrieval Augmented Generation.

## What I learned building this
- How RAG pipelines work at the code level without abstractions
- How ChromaDB stores and searches embeddings using similarity
- How to build grounded LLM prompts that prevent hallucination
- How retrieval and generation work together in a complete pipeline

## Next Improvements
- PDF support instead of plain text file
- Streamlit UI for browser based interaction
- FastAPI backend to serve as an API endpoint
- Persistent ChromaDB storage across sessions
