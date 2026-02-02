import os
import sys
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import json

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "job_offers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# France Travail API Configuration
# Pour obtenir vos credentials: https://francetravail.io/
FRANCE_TRAVAIL_CLIENT_ID = os.getenv("FRANCE_TRAVAIL_CLIENT_ID", "")
FRANCE_TRAVAIL_CLIENT_SECRET = os.getenv("FRANCE_TRAVAIL_CLIENT_SECRET", "")
FRANCE_TRAVAIL_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FRANCE_TRAVAIL_TOKEN_URL = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"


def get_access_token():
    """Obtenir un token d'accès pour l'API France Travail."""
    if not FRANCE_TRAVAIL_CLIENT_ID or not FRANCE_TRAVAIL_CLIENT_SECRET:
        sys.exit(1)
    
    try:
        response = requests.post(
            FRANCE_TRAVAIL_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": FRANCE_TRAVAIL_CLIENT_ID,
                "client_secret": FRANCE_TRAVAIL_CLIENT_SECRET,
                "scope": "api_offresdemploiv2 o2dsoffre"
            }
        )
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]
    except Exception as e:
        sys.exit(1)

def fetch_job_offers(access_token, max_offers=100):
    """Récupérer les offres d'emploi depuis France Travail."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    
    all_offers = []
    range_start = 0
    range_size = 150  # Maximum par requête
    
    while len(all_offers) < max_offers:
        try:
            params = {
                "range": f"{range_start}-{range_start + range_size - 1}",
                "sort": "1"  # Tri par date de création décroissante
            }
            
            response = requests.get(
                FRANCE_TRAVAIL_API_URL,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            resultats = data.get("resultats", [])
            if not resultats:
                break
            
            all_offers.extend(resultats)
            range_start += range_size
            
            if len(all_offers) >= max_offers:
                all_offers = all_offers[:max_offers]
                break
                
            # Pause pour respecter les limites de l'API
            time.sleep(0.5)
            
        except Exception as e:
            break
    
    return all_offers

def format_job_offer(offer):
    """Formater une offre d'emploi en texte pour l'embedding."""
    # Extraction des informations clés
    intitule = offer.get("intitule", "")
    description = offer.get("description", "")
    lieu = offer.get("lieuTravail", {})
    ville = lieu.get("libelle", "")
    entreprise = offer.get("entreprise", {}).get("nom", "Non spécifié")
    type_contrat = offer.get("typeContrat", "")
    experience = offer.get("experienceExige", "")
    competences = offer.get("competences", [])
    
    # Construction du texte
    text_parts = [
        f"Poste: {intitule}",
        f"Entreprise: {entreprise}",
        f"Lieu: {ville}",
        f"Type de contrat: {type_contrat}",
        f"Expérience: {experience}",
    ]
    
    if competences:
        comp_list = [c.get("libelle", "") for c in competences]
        text_parts.append(f"Compétences: {', '.join(comp_list)}")
    
    if description:
        text_parts.append(f"\nDescription:\n{description}")
    
    return "\n".join(text_parts)

def process_job_offers(offers):
    """Traiter les offres d'emploi et créer les documents pour l'indexation."""
    all_documents = []
    
    for offer in tqdm(offers, desc="Traitement des offres"):
        try:
            text = format_job_offer(offer)
            
            if text.strip():
                all_documents.append({
                    "text": text,
                    "offer_id": offer.get("id", ""),
                    "intitule": offer.get("intitule", ""),
                    "entreprise": offer.get("entreprise", {}).get("nom", "Non spécifié"),
                    "lieu": offer.get("lieuTravail", {}).get("libelle", ""),
                    "type_contrat": offer.get("typeContrat", ""),
                    "date_creation": offer.get("dateCreation", ""),
                    "url_postuler": offer.get("origineOffre", {}).get("urlOrigine", "")
                })
        except Exception as e:
            continue
    
    return all_documents

def create_collection(client, embedding_dim):
    """Create or recreate the Qdrant collection."""
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )

def index_documents(documents, client, model):
    """Index job offer documents into Qdrant."""
    batch_size = 100
    points = []
    
    for idx, doc_data in enumerate(tqdm(documents, desc="Création des embeddings")):
        # Generate embedding
        embedding = model.encode(doc_data["text"]).tolist()
        
        # Create point with metadata
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "text": doc_data["text"],
                "offer_id": doc_data.get("offer_id", ""),
                "intitule": doc_data.get("intitule", ""),
                "entreprise": doc_data.get("entreprise", ""),
                "lieu": doc_data.get("lieu", ""),
                "type_contrat": doc_data.get("type_contrat", ""),
                "date_creation": doc_data.get("date_creation", ""),
                "url_postuler": doc_data.get("url_postuler", "")
            }
        )
        points.append(point)
        
        # Upload en batch
        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    
    # Upload les points restants
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

def main():
    # Attendre la disponibilité de Qdrant
    max_retries = 30
    client = None
    for i in range(max_retries):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            break
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                sys.exit(1)
    
    # Check if collection already exists and has documents
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        points_count = collection_info.points_count
        if points_count > 0:
            sys.exit(0)
    except Exception:
        pass
    
    # Charger le modèle d'embedding
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # Obtenir le token d'accès
    access_token = get_access_token()
    
    # Récupérer les offres d'emploi
    max_offers = int(os.getenv("MAX_JOB_OFFERS", "500"))
    offers = fetch_job_offers(access_token, max_offers=max_offers)
    
    if not offers:
        return
    
    # Procéder aux offres
    documents = process_job_offers(offers)
    
    if not documents:
        return
    
    # Créer la collection Qdrant
    create_collection(client, embedding_dim)
    
    # Index documents
    index_documents(documents, client, model)

if __name__ == "__main__":
    main()
