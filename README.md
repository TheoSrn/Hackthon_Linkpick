# Système RAG pour Documents PDF

Un système complet de Retrieval-Augmented Generation (RAG) permettant d'interroger vos documents PDF en langage naturel.

## Fonctionnalités

- **Indexation automatique** des documents PDF
- **Recherche sémantique** dans vos documents
- **Deux modes de recherche** :
  - Mode LLM : Génération de réponses contextuelles avec vLLM
  - Mode Keywords : Recherche rapide par mots-clés sans LLM
- **Génération de réponses** basées sur le contenu des PDFs
- **Interface web** intuitive
- **Déploiement facile** avec Docker Compose

## Prérequis

- Docker et Docker Compose installés
- GPU NVIDIA (recommandé) ou CPU
- Au moins 8 GB de RAM (16 GB recommandé)
- Documents PDF dans le dossier `dataset/`

## Installation et Démarrage

### 1. Ajouter vos documents PDF

Placez vos fichiers PDF dans le dossier `dataset/` :

```bash
cp /chemin/vers/vos/pdfs/*.pdf dataset/
```

### 2. Lancer le système

```bash
# Démarrer tous les services
docker compose up -d

# Vérifier le statut
docker compose ps
```

Le système va :
1. Démarrer Qdrant (base de données vectorielle)
2. Indexer vos documents PDF
3. Démarrer vLLM (serveur d'inférence)
4. Lancer l'API backend
5. Lancer l'interface web

**Note :** Le premier démarrage peut prendre 5-10 minutes (téléchargement du modèle).

### 3. Accéder au système

- **Interface Web** : http://localhost
- **API Backend** : http://localhost:8001
- **Dashboard Qdrant** : http://localhost:6333/dashboard

## Exemple d'Utilisation

### Via l'Interface Web

1. Ouvrez http://localhost dans votre navigateur
2. Attendez que le statut système indique "OK"
3. Choisissez le mode de recherche :
   - **With LLM** : Génère une réponse complète basée sur les documents
   - **Keywords Only** : Recherche rapide par mots-clés sans génération LLM
4. Tapez votre question ou vos mots-clés dans le champ de texte
5. Cliquez sur "Ask Question" (mode LLM) ou "Search" (mode Keywords)
6. Consultez la réponse (mode LLM) et les sources citées

**Exemple de question (mode LLM) :**
```
Quels sont les principaux sujets abordés dans ces documents ?
```

**Exemple de mots-clés (mode Keywords) :**
```
machine learning, neural networks, deep learning
```

### Via l'API (cURL)

**Mode LLM (avec génération de réponse) :**

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelles sont les principales conclusions ?",
    "top_k": 3
  }'
```

**Réponse :**
```json
{
  "answer": "D'après les documents, les principales conclusions sont...",
  "sources": [
    {
      "filename": "document.pdf",
      "chunk_id": 5,
      "score": 0.89,
      "text": "Extrait du texte pertinent..."
    }
  ]
}
```

**Mode Keywords (recherche rapide sans LLM) :**

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "machine learning neural networks",
    "top_k": 5
  }'
```

**Réponse :**
```json
{
  "sources": [
    {
      "filename": "document.pdf",
      "chunk_id": 12,
      "score": 0.92,
      "text": "Texte complet du chunk pertinent..."
    }
  ]
}
```

### Via l'API (Python)

**Mode LLM :**

```python
import requests

response = requests.post(
    "http://localhost:8001/query",
    json={
        "question": "Résumez les points clés du document",
        "top_k": 3
    }
)

result = response.json()
print("Réponse:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['filename']} (Score: {source['score']:.2f})")
```

**Mode Keywords :**

```python
import requests

response = requests.post(
    "http://localhost:8001/search",
    json={
        "keywords": "artificial intelligence deep learning",
        "top_k": 5
    }
)

result = response.json()
print(f"Trouvé {len(result['sources'])} résultats\n")
for i, source in enumerate(result["sources"], 1):
    print(f"{i}. {source['filename']} - Score: {source['score']:.2f}")
    print(f"   {source['text'][:100]}...\n")
```

## Architecture

```
┌─────────────┐
│  Frontend   │  ← Interface web (Nginx)
│ (Port 80)   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Backend   │  ← API FastAPI
│ (Port 8001) │
└──────┬──────┘
       │
    ┌──▼───────────┬──────────┐
    │              │          │
┌───▼────┐  ┌─────▼─────┐  ┌─▼────┐
│ Qdrant │  │   vLLM    │  │Indexer│
│ (6333) │  │  (8000)   │  │       │
└────────┘  └───────────┘  └───────┘
```

## Configuration

### Modifier le nombre de chunks récupérés

Dans `backend/main.py` :
```python
TOP_K = 3  # Nombre de chunks à récupérer
```

### Changer la taille des chunks

Dans `indexer/index_documents.py` :
```python
CHUNK_SIZE = 500      # Caractères par chunk
CHUNK_OVERLAP = 50    # Chevauchement entre chunks
```

### Utiliser un modèle différent

Dans `docker-compose.yml` :
```yaml
vllm:
  command: >
    --model microsoft/Phi-3-mini-4k-instruct
    --dtype float16
```

## Statistiques du Système

Consultez les statistiques d'indexation :

```bash
curl http://localhost:8001/stats
```

Réponse :
```json
{
  "collection_name": "pdf_documents",
  "total_chunks": 252,
  "vector_size": 384
}
```

## Commandes Utiles

```bash
# Voir les logs
docker compose logs -f

# Redémarrer un service
docker compose restart backend

# Arrêter le système
docker compose down

# Re-indexer les documents
docker compose up -d indexer

# Vérifier l'état des conteneurs
docker compose ps
```

## Dépannage

### Le système ne démarre pas

Vérifiez que les ports ne sont pas déjà utilisés :
```bash
lsof -i :80,8001,6333,8000
```

### Pas de résultats trouvés

Vérifiez que les documents sont indexés :
```bash
curl http://localhost:8001/stats
```

### vLLM prend trop de temps

Utilisez un modèle plus petit ou passez en mode CPU :
```bash
docker compose -f docker-compose.cpu.yml up -d
```

## Structure du Projet

```
othman_zyad/
├── dataset/              # Vos fichiers PDF
├── indexer/              # Service d'indexation
│   ├── Dockerfile
│   ├── requirements.txt
│   └── index_documents.py
├── backend/              # API FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── frontend/             # Interface web
│   ├── Dockerfile
│   └── index.html
├── docker-compose.yml    # Configuration Docker
└── README.md
```
