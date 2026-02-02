# Système RAG pour Offres d'Emploi France Travail

Un système complet de Retrieval-Augmented Generation (RAG) permettant de rechercher et d'interroger les offres d'emploi de France Travail (anciennement Pôle Emploi) en langage naturel.

## Fonctionnalités

- **Récupération automatique** des offres d'emploi depuis l'API open data France Travail
- **Indexation intelligente** des offres dans une base de données vectorielle (Qdrant)
- **Recherche sémantique** avec embeddings (all-MiniLM-L6-v2)
- **Génération de réponses** contextuelles avec vLLM et Qwen2.5-1.5B-Instruct
- **Réponses formatées en Markdown** (listes, gras, structure claire)
- **Interface web française** intuitive et épurée
- **Déploiement facile** avec Docker Compose
- **Accélération GPU** avec support NVIDIA CUDA

## Prérequis

- Docker et Docker Compose installés
- **GPU NVIDIA** (recommandé, 8GB VRAM minimum) avec drivers CUDA
  - Testé sur RTX 4060 Laptop (8GB VRAM)
- Au moins 16 GB de RAM système
- **Credentials API France Travail** (voir section Configuration)

## Configuration de l'API France Travail

### 1. Obtenir vos credentials

1. Rendez-vous sur [France Travail Connect](https://francetravail.io/)
2. Créez un compte développeur
3. Inscrivez-vous à l'API "Offres d'emploi v2"
4. Récupérez votre `client_id` et `client_secret`

### 2. Configurer les variables d'environnement

Copiez le fichier `.env.example` en `.env` :

```bash
cp .env.example .env
```

Éditez le fichier `.env` et ajoutez vos credentials :

```env
FRANCE_TRAVAIL_CLIENT_ID=votre_client_id
FRANCE_TRAVAIL_CLIENT_SECRET=votre_client_secret
MAX_JOB_OFFERS=500
```

## Installation et Démarrage

### 1. Lancer le système

```bash
# Démarrer tous les services
docker compose up -d

# Vérifier le statut
docker compose ps
```

Le système va :
1. Démarrer Qdrant (base de données vectorielle)
2. Lancer l'indexeur pour récupérer et indexer les offres France Travail
   - L'indexeur vérifie si la collection existe et contient déjà des données
   - Si oui, il quitte immédiatement (pas de réindexation)
   - Si non, il récupère jusqu'à 500 offres et les indexe
3. Démarrer vLLM (serveur d'inférence avec Qwen2.5-1.5B-Instruct)
   - Attend que l'indexeur termine
   - Charge le modèle sur GPU
4. Lancer l'API backend FastAPI
5. Lancer l'interface web Nginx

**Note :** Le premier démarrage peut prendre 10-15 minutes (téléchargement du modèle Qwen ~3GB et initialisation GPU).

### 3. Accéder au système

- **Interface Web** : http://localhost
- **API Backend** : http://localhost:8001
- **Dashboard Qdrant** : http://localhost:6333/dashboard

## Exemple d'Utilisation

### Via l'Interface Web

1. Ouvrez http://localhost dans votre navigateur
2. Attendez que le statut système indique "✓ Système prêt"
3. Tapez votre question en français dans le champ de texte
4. Cliquez sur "Rechercher"
5. Consultez la réponse formatée en Markdown avec :
   - Titres en **gras**
   - Listes numérotées pour les offres
   - Structure claire et lisible
6. Examinez les sources citées (format "Intitulé - Entreprise")

**Exemples de questions :**
```
Quelles sont les offres de développeur Python disponibles ?

Liste-moi les offres en CDI dans le secteur tech

Trouve-moi des postes de Data Scientist à Paris
```

### Via l'API (cURL)

**Recherche avec génération de réponse :**

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelles sont les offres de développeur disponibles ?",
    "top_k": 3
  }'
```

**Réponse :**
```json
{
  "answer": "Voici les offres de développeur trouvées :\n\n1. **Développeur Full Stack** chez TechCorp\n...",
  "sources": [
    {
      "intitule": "Développeur Full Stack",
      "entreprise": "TechCorp",
      "chunk_id": 5,
      "score": 0.89,
      "text": "Extrait du texte de l'offre..."
    }
  ]
}
```

### Via l'API (Python)

```python
import requests

response = requests.post(
    "http://localhost:8001/query",
    json={
        "question": "Liste les offres de Data Scientist",
        "top_k": 3
    }
)

result = response.json()
print("Réponse:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['intitule']} chez {source['entreprise']} (Score: {source['score']:.2f})")
```

## Architecture

```
┌─────────────┐
│  Frontend   │  ← Interface web française (Nginx + Markdown)
│ (Port 80)   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Backend   │  ← API FastAPI (prompts français)
│ (Port 8001) │
└──────┬──────┘
       │
    ┌──▼───────────┬──────────┐
    │              │          │
┌───▼────┐  ┌─────▼─────┐  ┌─▼────────┐
│ Qdrant │  │   vLLM    │  │ Indexer  │
│ (6333) │  │  (8000)   │  │(one-shot)│
│        │  │  Qwen2.5  │  │          │
└────────┘  └───────────┘  └──────────┘
   ↑              ↑
   │              │
   └──────────────┘
     France Travail API
```

## Spécifications Techniques

### Modèles et Embeddings
- **LLM** : Qwen/Qwen2.5-1.5B-Instruct (1.5B paramètres)
  - Context window : 4096 tokens
  - Format : float16
  - GPU memory utilization : 70%
- **Embeddings** : sentence-transformers/all-MiniLM-L6-v2
  - Dimension : 384
  - Utilisé pour la recherche sémantique

### Base de Données
- **Qdrant** : Base vectorielle pour stocker les offres
- **Collection** : "job_offers"
- **Champs par offre** :
  - `text` : Description complète
  - `offer_id` : ID unique France Travail
  - `intitule` : Titre du poste
  - `entreprise` : Nom de l'entreprise
  - `lieu` : Localisation
  - `type_contrat` : Type de contrat (CDI, CDD, etc.)
  - `date_creation` : Date de publication
  - `url_postuler` : Lien pour postuler

### Configuration GPU
- CUDA graphs activés
- FlashAttention backend
- Memory utilization : ~7.5GB sur 8GB VRAM

## Configuration

### Modifier le nombre de résultats

Dans `backend/main.py` :
```python
TOP_K = 3  # Nombre d'offres à récupérer pour le contexte
```

### Modifier le nombre d'offres indexées

Dans le fichier `.env` :
```env
MAX_JOB_OFFERS=500  # Nombre maximum d'offres à indexer
```

### Utiliser un modèle différent

Dans `docker-compose.yml`, section vLLM :
```yaml
vllm:
  command: >
    --model Qwen/Qwen2.5-1.5B-Instruct
    --max-model-len 4096
    --dtype float16
    --gpu-memory-utilization 0.7
```

Modèles compatibles recommandés :
- `Qwen/Qwen2.5-1.5B-Instruct` (actuel, 4096 tokens)
- `Qwen/Qwen2.5-3B-Instruct` (plus performant, nécessite plus de VRAM)
- `microsoft/Phi-3-mini-4k-instruct` (alternatif)

### Ajuster la longueur du contexte

Modifier `--max-model-len` dans docker-compose.yml :
```yaml
--max-model-len 4096  # Peut aller jusqu'à 32768 pour Qwen2.5
```

## Statistiques du Système

Consultez les statistiques d'indexation :

```bash
curl http://localhost:8001/stats
```

Réponse :
```json
{
  "collection_name": "job_offers",
  "total_chunks": 500,
  "vector_size": 384
}
```

## Commandes Utiles

```bash
# Voir les logs en temps réel
docker compose logs -f

# Logs d'un service spécifique
docker compose logs -f backend
docker compose logs -f vllm

# Redémarrer un service
docker compose restart backend

# Arrêter le système
docker compose down

# Arrêter et supprimer les volumes (réinitialisation complète)
docker compose down -v

# Re-indexer les offres (supprime et recrée la collection)
docker compose down qdrant indexer
docker compose up -d qdrant indexer

# Vérifier l'état des conteneurs
docker compose ps

# Vérifier l'utilisation GPU
nvidia-smi

# Rebuild un service après modification
docker compose up -d --build backend
```

## Dépannage

### Le système ne démarre pas

Vérifiez que les ports ne sont pas déjà utilisés :
```bash
lsof -i :80,8001,6333,8000
```

### Pas de résultats trouvés

Vérifiez que les offres sont indexées :
```bash
curl http://localhost:8001/stats
# Doit afficher total_chunks > 0
```

Si vide, réindexer :
```bash
docker compose restart indexer
docker compose logs -f indexer
```

### vLLM ne démarre pas / Out of Memory

Le modèle Qwen2.5-1.5B nécessite ~7.5GB VRAM. Solutions :

1. **Réduire la mémoire utilisée** (docker-compose.yml) :
```yaml
--gpu-memory-utilization 0.6  # au lieu de 0.7
```

2. **Réduire le contexte** :
```yaml
--max-model-len 2048  # au lieu de 4096
```

3. **Vérifier CUDA** :
```bash
nvidia-smi  # Vérifier GPU disponible
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Les réponses sont lentes

- **Normal** : Première requête prend ~30s (warmup)
- **Après** : ~5-10s par requête sur RTX 4060 Laptop
- Pour accélérer : Utiliser GPU plus puissant ou modèle plus petit

### L'interface ne charge pas

Vérifier que tous les services sont UP :
```bash
docker compose ps
# Tous doivent être "running" et "healthy"
```

Vérifier les logs du backend :
```bash
docker compose logs backend
```

### Credentials France Travail invalides

Erreur : `401 Unauthorized` dans les logs indexer

Solution :
1. Vérifier `.env` avec les bons credentials
2. Redemander l'accès sur [France Travail Connect](https://francetravail.io/)
3. Vérifier que l'API "Offres d'emploi v2" est bien souscrite

### Le Markdown ne s'affiche pas

Vérifier que `marked.js` est bien chargé dans le frontend :
```bash
curl http://localhost/index.html | grep marked
# Doit trouver: <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js">
```

## Structure du Projet

```
Hackthon_Linkpick/
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
