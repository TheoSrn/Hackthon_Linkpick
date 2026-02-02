from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VLLM_HOST = os.getenv("VLLM_HOST", "vllm")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3 

# Initialize FastAPI
app = FastAPI(title="RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for clients
qdrant_client = None
embedding_model = None
vllm_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup."""
    global qdrant_client, embedding_model, vllm_client
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")
    
    # Initialize vLLM client
    vllm_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
    )
    print(f"Connected to vLLM at {VLLM_HOST}:{VLLM_PORT}")

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class KeywordSearchRequest(BaseModel):
    keywords: str
    top_k: int = TOP_K

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

class KeywordSearchResponse(BaseModel):
    sources: list[dict]

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG API"}

@app.get("/health")
async def health():
    """Detailed health check."""
    try:
        # Check Qdrant
        collections = qdrant_client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    try:
        # Check vLLM
        models = vllm_client.models.list()
        vllm_status = "healthy"
    except Exception as e:
        vllm_status = f"unhealthy: {str(e)}"
    
    return {
        "qdrant": qdrant_status,
        "vllm": vllm_status,
        "embedding_model": EMBEDDING_MODEL
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    1. Embed the question
    2. Search for relevant chunks in Qdrant
    3. Feed chunks to vLLM for answer generation
    """
    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode(request.question).tolist()
        
        # Search for relevant chunks using query method
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_embedding,
            limit=request.top_k
        ).points
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Prepare context from retrieved chunks
        context_parts = []
        sources = []
        
        for idx, result in enumerate(search_results):
            chunk_text = result.payload.get("text", "")
            filename = result.payload.get("filename", "unknown")
            chunk_id = result.payload.get("chunk_id", 0)
            score = result.score
            
            context_parts.append(f"[Document {idx+1}: {filename}, Chunk {chunk_id}]\n{chunk_text}")
            sources.append({
                "filename": filename,
                "chunk_id": chunk_id,
                "score": float(score),
                "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for vLLM
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Question: {request.question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite which document(s) you're using in your answer

Answer:"""
        
        # Query vLLM
        completion = vllm_client.chat.completions.create(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = completion.choices[0].message.content
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search", response_model=KeywordSearchResponse)
async def keyword_search(request: KeywordSearchRequest):
    """
    Search for documents by keywords without LLM generation.
    
    1. Embed the keywords
    2. Search for relevant chunks in Qdrant
    3. Return the top K results
    """
    try:
        # Generate embedding for the keywords
        keywords_embedding = embedding_model.encode(request.keywords).tolist()
        
        # Search for relevant chunks using query method
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=keywords_embedding,
            limit=request.top_k
        ).points
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Prepare sources from retrieved chunks
        sources = []
        
        for result in search_results:
            chunk_text = result.payload.get("text", "")
            filename = result.payload.get("filename", "unknown")
            chunk_id = result.payload.get("chunk_id", 0)
            score = result.score
            
            sources.append({
                "filename": filename,
                "chunk_id": chunk_id,
                "score": float(score),
                "text": chunk_text
            })
        
        return KeywordSearchResponse(sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed documents."""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
