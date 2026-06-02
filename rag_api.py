from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from docx import Document
from groq import Groq
import chromadb
import io
import os

app = FastAPI(title="DocMind RAG API")

# Allow HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontend
app.mount("/static", StaticFiles(directory="."), name="static")

# In memory storage
chroma_client = chromadb.Client()
collection_store = {}

# ── Models ────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    top_k: int = 3

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("docmind.html")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        doc = Document(io.BytesIO(contents))

        # Extract text
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + ". "

        # Split into chunks
        chunks = []
        for sentence in text.split("."):
            sentence = sentence.strip()
            if len(sentence) > 20:
                chunks.append(sentence)

        if not chunks:
            return {"error": "No text could be extracted from document"}

        # Store in ChromaDB
        collection_name = "current_doc"

        # Delete existing collection if exists
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass

        collection = chroma_client.create_collection(collection_name)
        collection.add(
            documents=chunks,
            ids=[str(i) for i in range(len(chunks))]
        )

        collection_store["current"] = collection

        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Document processed successfully — {len(chunks)} chunks stored"
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if "current" not in collection_store:
            return {"error": "No document uploaded yet. Please upload a document first."}

        collection = collection_store["current"]

        # Retrieve relevant chunks
        results = collection.query(
            query_texts=[request.question],
            n_results=request.top_k
        )
        relevant_chunks = results["documents"][0]
        context = "\n".join(relevant_chunks)

        # Build grounded prompt
        prompt = f"""You are a helpful document assistant named DocMind.
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have that information in the document."
Be concise, precise, and helpful.

Context:
{context}

Question: {request.question}
Answer:"""

        # Call Groq
        groq_client = Groq(api_key=request.api_key)
        response = groq_client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return {
            "success": True,
            "answer": answer,
            "sources": relevant_chunks
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "DocMind API is running"}
