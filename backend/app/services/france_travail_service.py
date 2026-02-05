"""Service de l'API France Travail."""
import json
import re
import requests
from fastapi import HTTPException
from app.config import (
    FRANCE_TRAVAIL_CLIENT_ID,
    FRANCE_TRAVAIL_CLIENT_SECRET,
    FRANCE_TRAVAIL_API_URL,
    FRANCE_TRAVAIL_TOKEN_URL
)
from app.services.llm_service import get_vllm_client


def get_access_token():
    """Obtenir le jeton d'accès pour l'API France Travail."""
    if not FRANCE_TRAVAIL_CLIENT_ID or not FRANCE_TRAVAIL_CLIENT_SECRET:
        raise HTTPException(
            status_code=500, 
            detail="France Travail API credentials not configured. Please set FRANCE_TRAVAIL_CLIENT_ID and FRANCE_TRAVAIL_CLIENT_SECRET environment variables."
        )
    
    try:
        response = requests.post(
            FRANCE_TRAVAIL_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": FRANCE_TRAVAIL_CLIENT_ID,
                "client_secret": FRANCE_TRAVAIL_CLIENT_SECRET,
                "scope": "api_offresdemploiv2 o2dsoffre"
            },
            timeout=10
        )
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting France Travail token: {str(e)}")


def extract_keywords_from_cv(cv_text: str) -> dict:
    """Extraire les mots-clés de recherche du CV en utilisant le LLM."""
    try:
        vllm_client = get_vllm_client()
        
        keyword_prompt = f"""Analyse ce CV et extrais les informations clés pour rechercher des offres d'emploi correspondantes.

CV :
{cv_text[:3000]}

Fournis en format JSON strict (sans markdown) :
{{
  "metier": "intitulé du poste recherché (max 50 caractères)",
  "competences": "2-3 compétences PRINCIPALES UNIQUEMENT séparées par des virgules"
}}

Exemples :
- metier: "Développeur Python", "Ingénieur Data", "Chef de projet IT"
- competences: "Python, Machine Learning, Docker" ou "Java, Spring, SQL"

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""
        
        completion = vllm_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": keyword_prompt}],
            temperature=0.2,
            max_tokens=200
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Essayer d'extraire le JSON de la réponse
        # Retirer les blocs de code markdown si présents
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        keywords_data = json.loads(response_text)
        return keywords_data
    except Exception as e:
        # Repli: retourner une recherche simple à partir des premiers mots du CV
        print(f"Error extracting keywords with LLM: {e}")
        # Extraire les premiers mots significatifs du CV
        words = cv_text[:200].split()
        simple_search = " ".join(words[:10]) if len(words) > 10 else " ".join(words)
        return {
            "metier": simple_search[:50],
            "competences": ""
        }


def search_job_offers(keywords_data: dict, max_results: int = 10) -> list:
    """Rechercher des emplois sur l'API France Travail basé sur les mots-clés."""
    try:
        access_token = get_access_token()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        # Stratégie: Essayer plusieurs recherches du plus spécifique au plus général
        search_strategies = []
        
        metier = keywords_data.get("metier", "").strip()
        competences = keywords_data.get("competences", "").strip()
        skills_list = [s.strip() for s in competences.split(',')] if competences else []
        
        # Simplifier l'intitulé du poste - prendre seulement les 3 premiers mots
        simple_metier = " ".join(metier.split()[:3]) if metier else ""
        
        # Stratégie 1: Métier + 1-2 premières compétences (le plus spécifique)
        if simple_metier and skills_list:
            first_skills = " ".join(skills_list[:2])
            combined = f"{simple_metier} {first_skills}"
            search_strategies.append(combined[:50])
        
        # Stratégie 2: Compétences restantes (si nous en avons plus de 2)
        if len(skills_list) > 2:
            remaining_skills = " ".join(skills_list[2:])
            search_strategies.append(remaining_skills[:50])
        
        # Stratégie 3: Métier uniquement
        if simple_metier:
            search_strategies.append(simple_metier)
        
        # Stratégie 4: Première compétence uniquement
        if skills_list:
            search_strategies.append(skills_list[0])
        
        # Stratégie 5: Aucun mot-clé (obtenir les offres récentes)
        search_strategies.append("")
        
        # Essayer chaque stratégie jusqu'à obtenir des résultats
        for idx, motsCles in enumerate(search_strategies):
            params = {
                "range": f"0-{max_results-1}",
                "sort": "1"  # Trier par date de création décroissante
            }
            
            # Ajouter motsCles seulement si nous avons des mots-clés significatifs
            if motsCles and len(motsCles) > 3:
                params["motsCles"] = motsCles[:50]  # Garder court
            
            print(f"France Travail API Request (Strategy {idx+1}/{len(search_strategies)}) - Keywords: '{motsCles}'")
            
            response = requests.get(
                FRANCE_TRAVAIL_API_URL,
                headers=headers,
                params=params,
                timeout=15
            )
            
            print(f"France Travail API Response - Status: {response.status_code}")
            
            # Gérer 204 No Content - essayer la stratégie suivante
            if response.status_code == 204:
                print(f"No results for strategy {idx+1}, trying next...")
                continue
            
            # Vérifier si la réponse est bien du JSON
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                print(f"Warning: Response is not JSON. Content-Type: {content_type}")
                if idx < len(search_strategies) - 1:
                    continue  # Try next strategy
                else:
                    raise HTTPException(
                        status_code=502, 
                        detail=f"France Travail API returned non-JSON response. Status: {response.status_code}"
                    )
            
            response.raise_for_status()
            
            try:
                data = response.json()
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {je}")
                if idx < len(search_strategies) - 1:
                    continue  # Try next strategy
                else:
                    raise HTTPException(
                        status_code=502,
                        detail="France Travail API returned invalid JSON response"
                    )
            
            offers = data.get("resultats", [])
            print(f"Found {len(offers)} job offers with strategy {idx+1}")
            
            if offers:
                return offers
            # Si aucune offre avec cette stratégie, essayer la suivante
            print(f"Strategy {idx+1} returned 0 offers, trying next...")
        
        # Si nous avons essayé toutes les stratégies et rien trouvé
        print("All search strategies exhausted, no offers found")
        return []
        
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" | Response: {e.response.text[:200]}"
        raise HTTPException(status_code=500, detail=f"Error searching France Travail API: {error_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing France Travail search: {str(e)}")


def format_job_offers(job_offers: list) -> tuple[list[dict], list[str]]:
    """Formater les offres d'emploi pour la réponse et le contexte."""
    matching_offers = []
    context_parts = []
    
    for idx, offer in enumerate(job_offers):
        intitule = offer.get("intitule", "Non spécifié")
        description = offer.get("description", "")
        lieu_travail = offer.get("lieuTravail", {})
        lieu = lieu_travail.get("libelle", "Non spécifié")
        entreprise_info = offer.get("entreprise", {})
        entreprise = entreprise_info.get("nom", "Non spécifié") if entreprise_info else "Non spécifié"
        type_contrat_code = offer.get("typeContrat", "")
        type_contrat_label = offer.get("typeContratLibelle", type_contrat_code)
        origine_offre = offer.get("origineOffre", {})
        url_postuler = origine_offre.get("urlOrigine", "") if origine_offre else ""
        date_creation = offer.get("dateCreation", "")
        
        # Créer le contexte pour l'analyse par le LLM
        offer_context = f"""Intitulé: {intitule}
Entreprise: {entreprise}
Lieu: {lieu}
Type de contrat: {type_contrat_label}
Description: {description[:500]}"""
        context_parts.append(f"[Offre {idx+1}]\n{offer_context}")
        
        matching_offers.append({
            "intitule": intitule,
            "entreprise": entreprise,
            "lieu": lieu,
            "type_contrat": type_contrat_label,
            "url_postuler": url_postuler,
            "date_creation": date_creation,
            "score": 1.0 - (idx * 0.05),  # Score décroissant basé sur l'ordre
            "description": description[:300] + "..." if len(description) > 300 else description
        })
    
    return matching_offers, context_parts
