import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import extract_text_from_pdf, chunk_text
from huggingface_hub import InferenceClient
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Load API keys ---
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# --- Hugging Face Client (for generation) ---
client = InferenceClient(api_key=HF_API_KEY)

# --- Models ---
embedding_model = SentenceTransformer("amixh/sentence-embedding-model-InLegalBERT-2")
generation_model = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Pinecone Setup ---
pc = Pinecone(api_key=PINECONE_API_KEY)

if "legal-ai-index" not in pc.list_indexes().names():
    pc.create_index(
        name="legal-ai-index",
        dimension=768,   # embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index("legal-ai-index")

# --- FastAPI App ---
app = FastAPI(title="Legal AI Assistant")

# --- Allow frontend requests ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request model for questions ---
class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a legal document and store its embeddings in Pinecone"""
    text = extract_text_from_pdf(file.file)
    chunks = chunk_text(text, chunk_size=500)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append((f"chunk-{i}", embedding, {"text": chunk}))

    index.upsert(vectors)
    return {"message": "File processed & stored", "chunks": len(chunks)}

@app.post("/ask")
async def ask_question(payload: Question):
    """Ask a question about uploaded documents"""
    query = payload.question
    q_embedding = embedding_model.encode(query).tolist()

    results = index.query(vector=q_embedding, top_k=3, include_metadata=True)
    context = "\n".join([m["metadata"]["text"] for m in results["matches"]])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer simply in plain English:"

    response = client.chat_completion(
        model=generation_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    try:
        answer = response.choices[0].message["content"]
    except (TypeError, AttributeError, KeyError):
        answer = response.choices[0].message.content

    return JSONResponse({"answer": answer, "context": context})


