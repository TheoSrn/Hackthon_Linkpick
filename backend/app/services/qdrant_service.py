"""Service Qdrant pour les opérations de base de données vectorielle."""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL

# Clients globaux
qdrant_client = None
embedding_model = None


def initialize_qdrant_client():
    """Initialiser le client Qdrant."""
    global qdrant_client
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return qdrant_client


def initialize_embedding_model():
    """Initialiser le modèle d'embedding."""
    global embedding_model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")
    return embedding_model


def get_qdrant_client():
    """Obtenir l'instance du client Qdrant."""
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return qdrant_client


def get_embedding_model():
    """Obtenir l'instance du modèle d'embedding."""
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    return embedding_model


def search_similar_documents(query_text: str, top_k: int = 3) -> list:
    """Rechercher des documents similaires dans Qdrant."""
    client = get_qdrant_client()
    model = get_embedding_model()
    
    # Générer l'embedding
    query_embedding = model.encode(query_text).tolist()
    
    # Rechercher
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points
    
    return search_results


def get_collection_stats():
    """Obtenir les statistiques sur la collection Qdrant."""
    client = get_qdrant_client()
    collection_info = client.get_collection(COLLECTION_NAME)
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection_info.points_count,
        "vector_size": collection_info.config.params.vectors.size
    }
