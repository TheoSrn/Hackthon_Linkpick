"""Paramètres de configuration de l'application."""
import os

# Configuration Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "job_offers"

# Configuration vLLM
VLLM_HOST = os.getenv("VLLM_HOST", "vllm")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# Configuration du modèle d'embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configuration de la recherche
TOP_K = 3

# Configuration de l'API France Travail
FRANCE_TRAVAIL_CLIENT_ID = os.getenv("FRANCE_TRAVAIL_CLIENT_ID", "")
FRANCE_TRAVAIL_CLIENT_SECRET = os.getenv("FRANCE_TRAVAIL_CLIENT_SECRET", "")
FRANCE_TRAVAIL_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FRANCE_TRAVAIL_TOKEN_URL = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
