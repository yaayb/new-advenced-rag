# EvidenceLab

Atelier RAG orienté évaluation : backend FastAPI, frontend React + Vite + Tailwind, Qdrant, Docker Compose.
Sans LangChain ni LlamaIndex : chaque étape est du Python explicite derrière des interfaces `Protocol`.

## Fonctionnalités
- Import de PDF, DOCX, TXT, Markdown (50 Mo max) → découpage par phrases → SQLite (source de vérité).
- Modes de recherche : **BM25** (implémentation maison), **dense** (Qdrant ou mémoire), **hybride** (Reciprocal Rank Fusion).
- Chaque requête renvoie, par segment : score, rang, nombre de tokens (et rangs BM25 / dense), plus la latence de chaque étape.
- Générateurs : `extractive` (aucun modèle requis) ou `openai` (tout serveur compatible OpenAI, ex. Ollama).
- Banc d'évaluation sur `eval_data/` (8 documents, 16 questions) : Hit@k, Recall@k, MRR, nDCG@k, couverture de la réponse, tokens de contexte, latence moyenne/p95. Les exécutions sont enregistrées dans SQLite.

## Lancer avec Docker
```bash
cp .env.example .env
docker compose up --build
```
Interface : http://localhost:8080 · API : http://localhost:8000/docs · Qdrant : http://localhost:6333

Pour un vrai LLM : lancez Ollama sur la machine hôte puis mettez `EVIDENCELAB_LLM_BACKEND=openai` dans `.env`.

## Développement local
```bash
cd backend && python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Linux : .venv/bin/pip
EVIDENCELAB_EVAL_DATA_DIR=../eval_data uvicorn evidencelab.api:app --reload
cd frontend && npm install && npm run dev      # http://localhost:5173
```
Vérifications : `ruff check .` et `pytest` (backend) ; `npm run typecheck && npm run build` (frontend).

## Déploiement en production (VPS + HTTPS)
Prérequis : un serveur avec Docker, un nom de domaine dont l'enregistrement DNS A pointe vers le serveur, ports 80 et 443 ouverts.
```bash
git clone https://github.com/yaayb/new-advenced-rag.git && cd new-advenced-rag
cp .env.example .env
# éditez .env : DOMAIN=votre-domaine.tld  et  EVIDENCELAB_API_KEY=<clé générée>
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # pour générer la clé
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
```
- Seul Caddy est exposé (80/443) et obtient automatiquement le certificat HTTPS. Qdrant et le backend restent sur le réseau interne.
- L'accès est protégé par la clé API (en-tête `X-API-Key`) : l'interface demande la clé au premier chargement. `/api/health` reste public (supervision).
- Le backend s'exécute en utilisateur non-root, avec healthcheck et redémarrage automatique.
- Sauvegardes : les volumes `backend_data` (SQLite) et `qdrant_data` ; Qdrant est reconstructible depuis SQLite au redémarrage.
- Mise à jour : `git pull && docker compose -f docker-compose.prod.yml --env-file .env up -d --build`.

Non couvert : pas de limitation de débit (rate limiting), une seule clé partagée (pas de comptes utilisateurs), un seul processus backend. Pour un usage multi-utilisateurs ou à forte charge, ajoutez une vraie authentification et un limiteur au niveau du proxy.

## Brancher une API gratuite (LLM et embeddings)
Sans clé, le système fonctionne en mode « extractif » (renvoie les meilleures phrases). Pour des réponses rédigées et une vraie recherche sémantique, créez une clé gratuite chez un fournisseur et renseignez `.env` (la clé ne doit **jamais** être commitée) :

| Fournisseur | `EVIDENCELAB_LLM_BASE_URL` | Modèle exemple |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | un modèle finissant par `:free` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` |
| Ollama (local) | `http://host.docker.internal:11434/v1` | `llama3.2` |

```env
EVIDENCELAB_LLM_BACKEND=openai
EVIDENCELAB_LLM_API_KEY=<votre clé>
EVIDENCELAB_EMBEDDING_BACKEND=openai   # embeddings neuronaux (ex. Gemini, offre gratuite)
EVIDENCELAB_EMBEDDING_API_KEY=<votre clé>
```
Après avoir changé d'embeddings, ré-importez vos documents. Les offres gratuites ont des quotas et peuvent utiliser vos données pour l'entraînement : n'y envoyez pas de documents confidentiels.

**Robustesse :** si l'API du LLM est en panne ou hors quota, la réponse bascule automatiquement sur le mode extractif (avec un avertissement dans l'interface) ; si l'API d'embeddings est indisponible, la recherche hybride se limite à BM25. Le service ne plante pas. Ce comportement est couvert par des tests avec de faux serveurs, mais **aucun fournisseur réel n'a été testé** (aucune clé n'était disponible).

## Limites
- L'encodeur par défaut est un encodeur par hachage de traits (aucun modèle à télécharger), déterministe : une base lexicale, pas un encodeur neuronal. Implémentez `Embedder` pour en brancher un.
- Le benchmark fourni est petit, facile et en anglais : les trois modes y obtiennent un score parfait, il ne les départage donc pas. Ajoutez des données plus difficiles dans `eval_data/`.
- Les PDF scannés ne sont pas pris en charge (pas d'OCR).
- Les latences dépendent de votre machine ; aucune valeur n'est publiée ici.
