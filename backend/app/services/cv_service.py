"""Service de traitement des CV."""
from fastapi import HTTPException
from app.services.llm_service import generate_completion, get_vllm_client
from app.services.france_travail_service import extract_keywords_from_cv, search_job_offers, format_job_offers
from app.utils.file_extractors import extract_text_from_pdf, extract_text_from_docx


def extract_cv_text(file_content: bytes, filename: str) -> str:
    """Extraire le texte du fichier CV en fonction du type de fichier."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith('.docx') or filename_lower.endswith('.doc'):
        return extract_text_from_docx(file_content)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Format de fichier non supporté. Veuillez uploader un PDF ou DOCX."
        )


def analyze_cv_profile(cv_text: str) -> str:
    """Analyser le CV et générer un résumé du profil."""
    analysis_prompt = f"""Analyse ce CV et fournis un résumé concis du profil professionnel en 2-3 phrases maximum.
Identifie les compétences clés, le domaine d'expertise et le type de poste recherché.

CV :
{cv_text[:2000]}

Résumé du profil (2-3 phrases maximum) :"""
    
    return generate_completion(analysis_prompt, temperature=0.3, max_tokens=200).strip()


def generate_job_matching_analysis(profile_summary: str, context_parts: list[str]) -> str:
    """Générer une analyse de correspondance d'emploi avec le LLM."""
    context = "\n\n".join(context_parts[:5])  # Limiter aux 5 premières offres pour le contexte
    
    analysis_prompt = f"""Tu es un conseiller en recrutement. Analyse la correspondance entre ce profil candidat et les offres d'emploi trouvées sur France Travail.

Profil du candidat :
{profile_summary}

Offres d'emploi disponibles (Top {min(5, len(context_parts))}) :
{context}

Fournis une analyse détaillée en Markdown avec :
1. **Correspondance générale** : Évalue la compatibilité du profil avec les offres (2-3 phrases)
2. **Offres recommandées** : Liste les meilleures offres avec pour chacune :
   - Titre et entreprise
   - Pourquoi cette offre correspond au profil (1-2 phrases)
   - Points forts de la candidature
3. **Conseils** : 2-3 recommandations pour optimiser les candidatures

Formate avec **gras**, listes à puces (-) et numérotation (1., 2., 3.)

Analyse :"""
    
    return generate_completion(analysis_prompt, temperature=0.7, max_tokens=800)


def process_cv_for_job_matching(file_content: bytes, filename: str, top_k: int = 10) -> dict:
    """Traiter le CV et trouver les offres d'emploi correspondantes."""
    # Extraire le texte du CV
    cv_text = extract_cv_text(file_content, filename)
    
    if not cv_text or len(cv_text.strip()) < 50:
        raise HTTPException(
            status_code=400, 
            detail="Le CV semble vide ou trop court. Veuillez vérifier le fichier."
        )
    
    # Analyser le profil du CV
    profile_summary = analyze_cv_profile(cv_text)
    
    # Extraire les mots-clés du CV
    keywords_data = extract_keywords_from_cv(cv_text)
    
    # Rechercher sur l'API France Travail des emplois correspondants
    job_offers = search_job_offers(keywords_data, max_results=top_k)
    
    if not job_offers:
        return {
            "analysis": "Aucune offre correspondante trouvée sur France Travail pour ce profil. Essayez d'élargir vos critères de recherche.",
            "matching_offers": [],
            "profile_summary": profile_summary
        }
    
    # Formater les offres d'emploi
    matching_offers, context_parts = format_job_offers(job_offers)
    
    # Générer l'analyse
    analysis = generate_job_matching_analysis(profile_summary, context_parts)
    
    return {
        "analysis": analysis,
        "matching_offers": matching_offers,
        "profile_summary": profile_summary
    }
