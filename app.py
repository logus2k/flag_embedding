# app.py

"""
pip install FlagEmbedding
pip install fastapi
pip install uvicorn
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import json

app = FastAPI(title="Embedding + Reranker API")

# CORS setup (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy model loading ---
dense_model = None
reranker_model = None

def get_dense_model():
    global dense_model
    if dense_model is None:
        print("Loading dense embedding model...")
        model_path = "./models/bge-m3"
        dense_model = BGEM3FlagModel(model_path, use_fp16=True)
        print("Dense model loaded.")
    return dense_model

def get_reranker_model():
    global reranker_model
    if reranker_model is None:
        print("Loading reranker model...")
        model_path = "./models/bge-reranker-v2-m3"
        reranker_model = BGEM3FlagModel(model_path, use_fp16=True)
        print("Reranker model loaded.")
    return reranker_model

# --- Models for input/output ---

class EmbeddingRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 12
    max_length: Optional[int] = 8192

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "embed": "bge-m3",
            "reranker": "bge-reranker-base"
        }
    }

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        model = get_dense_model()

        output = model.encode(
            sentences=request.texts,
            batch_size=request.batch_size,
            max_length=request.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        embeddings = output['dense_vecs'].tolist()
        dimensions = len(embeddings[0]) if embeddings else 0

        return {
            "embeddings": embeddings,
            "dimensions": dimensions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query and documents are required")

    try:
        model = get_reranker_model()

        pairs = [(request.query, doc) for doc in request.documents]
        output = model.encode(
            sentence_pairs=pairs,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=False
        )

        scores = output['scores'].tolist()
        return { "scores": scores }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reranking documents: {str(e)}")


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
