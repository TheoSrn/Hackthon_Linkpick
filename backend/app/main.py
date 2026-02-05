"""Application FastAPI principale avec les routes."""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    QueryRequest, QueryResponse,
    KeywordSearchRequest, KeywordSearchResponse,
    CVAnalysisResponse
)
from app.config import COLLECTION_NAME, TOP_K
from app.services.qdrant_service import (
    initialize_qdrant_client,
    initialize_embedding_model,
    get_qdrant_client,
    search_similar_documents,
    get_collection_stats
)
from app.services.llm_service import initialize_vllm_client, generate_completion
from app.services.cv_service import process_cv_for_job_matching


# Initialiser FastAPI
app = FastAPI(title="RAG API", version="1.0.0")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialiser les clients au démarrage."""
    initialize_qdrant_client()
    initialize_embedding_model()
    initialize_vllm_client()


@app.get("/")
async def root():
    """Endpoint de vérification de santé."""
    return {"status": "healthy", "service": "RAG API"}


@app.get("/health")
async def health():
    """Vérification détaillée de santé."""
    try:
        # Vérifier Qdrant
        qdrant_client = get_qdrant_client()
        collections = qdrant_client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    try:
        # Vérifier vLLM
        from app.services.llm_service import get_vllm_client
        vllm_client = get_vllm_client()
        models = vllm_client.models.list()
        vllm_status = "healthy"
    except Exception as e:
        vllm_status = f"unhealthy: {str(e)}"
    
    from app.config import EMBEDDING_MODEL
    return {
        "qdrant": qdrant_status,
        "vllm": vllm_status,
        "embedding_model": EMBEDDING_MODEL
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Interroger le système RAG avec une question.
    
    1. Générer l'embedding de la question
    2. Rechercher les chunks pertinents dans Qdrant
    3. Envoyer les chunks à vLLM pour générer la réponse
    """
    try:
        # Rechercher les chunks pertinents
        search_results = search_similar_documents(request.question, request.top_k)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Préparer le contexte à partir des chunks récupérés
        context_parts = []
        sources = []
        
        for idx, result in enumerate(search_results):
            chunk_text = result.payload.get("text", "")
            intitule = result.payload.get("intitule", "unknown")
            entreprise = result.payload.get("entreprise", "Non spécifié")
            chunk_id = result.payload.get("chunk_id", 0)
            score = result.score
            
            context_parts.append(f"[Document {idx+1}: {intitule}, Chunk {chunk_id}]\n{chunk_text}")
            sources.append({
                "intitule": intitule,
                "entreprise": entreprise,
                "chunk_id": chunk_id,
                "score": float(score),
                "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for vLLM
        prompt = f"""Tu es un assistant utile qui répond aux questions en se basant sur le contexte fourni provenant des offres d'emploi.

Contexte des offres d'emploi :
{context}

Question : {request.question}

Instructions :
- Liste TOUTES les offres mentionnées dans le contexte ci-dessus, sans exception
- Pour chaque offre, indique l'intitulé exact, l'entreprise et le lieu
- Ne saute aucune offre présente dans le contexte
- Si tu listes des offres, assure-toi de mentionner TOUTES celles présentes dans le contexte
- Sois concis et précis
- Ne mens pas et n'invente pas d'informations qui ne sont pas dans le contexte
- Formate ta réponse en Markdown : utilise **gras** pour les titres importants, des listes numérotées (1., 2., 3.) ou à puces (-) pour les éléments, et des sauts de ligne pour la lisibilité

Réponse :"""
        
        # Generate answer
        answer = generate_completion(prompt, temperature=0.7, max_tokens=500)
        
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
        # Search for relevant chunks
        search_results = search_similar_documents(request.keywords, request.top_k)
        
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
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/upload-cv", response_model=CVAnalysisResponse)
async def upload_cv(file: UploadFile = File(...), top_k: int = 10):
    """
    Upload a CV (PDF or DOCX) and get matching job offers from France Travail API.
    
    1. Extract text from the CV
    2. Analyze the CV to extract keywords and profile
    3. Query France Travail API for matching jobs
    4. Return detailed matching offers with application information
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Process CV and get matching jobs
        result = process_cv_for_job_matching(file_content, file.filename, top_k)
        
        return CVAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
