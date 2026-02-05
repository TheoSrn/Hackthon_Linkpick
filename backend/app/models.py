"""Modèles Pydantic pour les requêtes et réponses de l'API."""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class KeywordSearchRequest(BaseModel):
    keywords: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class KeywordSearchResponse(BaseModel):
    sources: list[dict]


class CVAnalysisResponse(BaseModel):
    analysis: str
    matching_offers: list[dict]
    profile_summary: str
