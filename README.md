# Buscador de trámites de Uruguay — Recuperación de Información

Aplicación web de RI sobre el **Catálogo de trámites y servicios del
Estado** (datos abiertos de AGESIC / catalogodatos.gub.uy).

## Arquitectura

```
                 ┌─────────────────────────────────────────┐
                 │  Elasticsearch 8 (docker-compose)       │
 data/           │  índice `tramites`, analyzer español:   │
 tramites.csv ──►│  lowercase+asciifolding+stop+stemmer    │
                 └──────────────▲──────────────────────────┘
                                │ bulk (src/es_index.py)
   fallback:                    │
   data/index.pkl (BM25 local)  │
                 ┌──────────────┴───────────┐
                 │  FastAPI  src/api.py     │  GET /api/search
                 │  multi_match ponderado,  │  ?q=&organismo=&canal=&gratuito=
                 │  facetas, highlighting   │
                 └──────────────▲───────────┘
                                │ proxy /api (vite.config.js)
                 ┌──────────────┴───────────┐
                 │  React + Vite + Tailwind │  frontend/
                 └──────────────────────────┘
```

- **Elasticsearch**: BM25 nativo con analyzer español custom, facetas
  (organismo/canal) y highlighting. Si ES no está disponible, la API cae
  al índice BM25 local (`rank_bm25`) construido por `src/index.py`.
- **Tesauro del dominio** (`src/synonyms.py`): expansión de consultas
  por sinónimos ("cédula"→"documento de identidad", "licencia"→"libreta
  de conducir", etc.) vía filtro `synonym_graph` en el `search_analyzer`
  — query-time, sin reindexar. `python -m src.synonyms` lo aplica al
  índice vivo.
- **Búsqueda híbrida**: BM25 + kNN sobre embeddings
  (`intfloat/multilingual-e5-small`, 384 dims) fusionados con
  **Reciprocal Rank Fusion** (`rrf_fuse` en `src/api.py`).
- **Reranking**: cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` sobre el
  top-k (two-stage retrieval), activable con `do_rerank=true`.
- **Chatbot RAG** (`POST /api/chat`): recupera top-5 con búsqueda
  híbrida y genera respuesta con **Ollama** (`llama3.2`). Si Ollama no
  está disponible, responde de forma extractiva con el mejor resultado.
- **CLI** (`src/search.py`): sigue disponible para pruebas rápidas.

## Setup

```bash
# Backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.index          # índice BM25 local (fallback)

# Elasticsearch (requiere Docker Desktop corriendo)
docker compose up -d
.venv/bin/python -m src.es_index       # crea índice e ingesta el CSV

# Frontend
cd frontend && npm install
```

El CSV se descarga de
https://catalogodatos.gub.uy/dataset/agesic-guia-de-tramites
(recurso "Tramites y servicios", formato CSV) → `data/tramites.csv`.

## Uso

```bash
# Terminal 1: API
.venv/bin/uvicorn src.api:app --port 8000

# Terminal 2: frontend
cd frontend && npm run dev   # http://localhost:5173

# CLI (sin levantar nada)
.venv/bin/python -m src.search "sacar pasaporte"
```

## API

- `GET /api/health` → estado y backend activo (`elasticsearch` | `bm25`)
- `GET /api/search?q=...&top_k=10&organismo=...&canal=internet&gratuito=true`
  → `{ total, results[], facets{organismos, canales}, backend }`

## Próximos pasos

- Búsqueda híbrida: BM25 + embeddings (`sentence-transformers` + kNN de ES).
- Reranking con cross-encoder multilingual.
- Corrección ortográfica y expansión de consultas.
- Evaluación con juicios de relevancia (precision@k, nDCG, `ir_measures`).
