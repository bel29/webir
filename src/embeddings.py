"""Embeddings densos para búsqueda semántica.

Modelo: intfloat/multilingual-e5-small (384 dims, buen español).
Convención e5: prefijar "passage: " a documentos y "query: " a consultas.

Uso:
    python -m src.embeddings          # genera y sube vectores a ES
"""

import sys
import time
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch, helpers

ES_URL = "http://localhost:9200"
INDEX = "tramites"
DATA_CSV = Path("data/tramites.csv")
MODEL_NAME = "intfloat/multilingual-e5-small"
DIMS = 384

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def doc_text(row: pd.Series) -> str:
    """Texto representativo del documento para el embedding."""
    parts = [row.get("nombre_tramite", ""), row.get("ques_es", "")]
    return "passage: " + " ".join(p for p in parts if p)[:1000]


def embed_query(query: str) -> list[float]:
    return get_model().encode("query: " + query, normalize_embeddings=True).tolist()


def embed_docs(texts: list[str]) -> list[list[float]]:
    return get_model().encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    ).tolist()


def main() -> None:
    es = Elasticsearch(ES_URL)
    if not es.ping():
        sys.exit("No se pudo conectar a Elasticsearch en " + ES_URL)

    # Agregar campo dense_vector al mapping existente.
    es.indices.put_mapping(
        index=INDEX,
        body={
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": DIMS,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        },
    )
    print(f"Mapping actualizado: campo 'embedding' ({DIMS} dims, cosine)")

    df = pd.read_csv(DATA_CSV, dtype=str, keep_default_na=False)
    texts = [doc_text(row) for _, row in df.iterrows()]

    t0 = time.time()
    vectors = embed_docs(texts)
    print(f"{len(vectors)} embeddings en {time.time() - t0:.1f}s")

    actions = (
        {
            "_op_type": "update",
            "_index": INDEX,
            "_id": row["id"],
            "doc": {"embedding": vec},
        }
        for (_, row), vec in zip(df.iterrows(), vectors)
    )
    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    print(f"Update: {success} docs, {len(errors)} errores")


if __name__ == "__main__":
    main()
