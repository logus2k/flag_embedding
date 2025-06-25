# app.py

import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# Define global variables for model paths
DENSE_MODEL_PATH = "/flagembedding/data/models/bge-m3"
RERANKER_MODEL_PATH = "/flagembedding/data/models/bge-reranker-v2-m3"

app = FastAPI(title="Embedding + Reranker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dense_model = None
reranker_model = None

def get_dense_model():
    global dense_model
    if dense_model is None:
        print("Loading dense embedding model...")
        dense_model = BGEM3FlagModel(DENSE_MODEL_PATH, use_fp16=True)
        print("Dense model loaded.")
    return dense_model

def get_reranker_model():
    global reranker_model
    if reranker_model is None:
        print("Loading reranker model...")
        reranker_model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)
        print("Reranker model loaded.")
    return reranker_model


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


@app.get("/health")
def health_check():
    status = "healthy"
    error_details = {}
    
    models_status = {
        "embed": {
            "name": "bge-m3", 
            "status": "unknown",
            "path": DENSE_MODEL_PATH
        },
        "reranker": {
            "name": "bge-reranker-v2-m3", 
            "status": "unknown",
            "path": RERANKER_MODEL_PATH
        }
    }
    
    # Check embedding model
    try:
        embed_model = get_dense_model()
        # Simple test to confirm the model works
        test_embedding = embed_model.encode(["Test health check"], return_dense=True)
        if test_embedding and "dense_vecs" in test_embedding:
            models_status["embed"]["status"] = "operational"
        else:
            models_status["embed"]["status"] = "loaded but test failed"
            status = "degraded"
    except Exception as e:
        models_status["embed"]["status"] = "failed"
        error_details["embed_error"] = str(e)
        status = "unhealthy"
    
    # Check reranker model
    try:
        rerank_model = get_reranker_model()
        # Simple test to confirm the model works
        test_scores = rerank_model.compute_score([("Test query", "Test document")])
        if test_scores and len(test_scores) > 0:
            models_status["reranker"]["status"] = "operational"
        else:
            models_status["reranker"]["status"] = "loaded but test failed"
            status = "degraded"
    except Exception as e:
        models_status["reranker"]["status"] = "failed"
        error_details["rerank_error"] = str(e)
        status = "unhealthy"
    
    response = {
        "status": status,
        "models": models_status,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    if error_details:
        response["errors"] = error_details
    
    return response


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

        embeddings = output['dense_vecs'].tolist() # type: ignore
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
        scores = model.compute_score(pairs)

        return { "scores": scores }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reranking documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
