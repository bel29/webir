"""Ingesta del catálogo de trámites en Elasticsearch.

Crea el índice `tramites` con un analyzer español custom
(lowercase + asciifolding + stopwords + stemming) y carga
los documentos vía bulk API.

Uso:
    python -m src.es_index            # crea índice e ingesta
    python -m src.es_index --recreate # borra y recrea el índice
"""

import sys
import time
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch, helpers

from src.synonyms import SEARCH_ANALYZER, syn_filter_def

ES_URL = "http://localhost:9200"
INDEX = "tramites"
DATA_CSV = Path("data/tramites.csv")

# Campos de texto completo (analizados en español).
TEXT_FIELDS = [
    "nombre_tramite",
    "ques_es",
    "casuistica",
    "requisitos_generales",
    "otros_datos_de_interes",
    "internet_como_se_hace",
    "persona_como_se_hace",
]

# Campos de texto que además sirven como faceta (keyword).
FACET_TEXT_FIELDS = [
    "institucion_nombre",
    "institucion_padre_organizacional_nombre",
    "categorias_de_organismo",
]

# Campos keyword puros.
KEYWORD_FIELDS = [
    "id",
    "url",
    "internet_url_del_tramite",
    "email_de_consulta",
    "unidad",
    "valor",
]

INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "spanish_custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "spanish_stop",
                        "spanish_stemmer",
                    ],
                },
                # Igual que spanish_custom pero con expansión por tesauro
                # del dominio — solo usable a tiempo de búsqueda.
                SEARCH_ANALYZER: {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        # domain_synonyms antes del stop: las reglas
                        # contienen stopwords ("de", "por").
                        "domain_synonyms",
                        "spanish_stop",
                        "spanish_stemmer",
                    ],
                },
            },
            "filter": {
                "spanish_stop": {"type": "stop", "stopwords": "_spanish_"},
                "spanish_stemmer": {
                    "type": "stemmer",
                    "language": "light_spanish",
                },
                "domain_synonyms": syn_filter_def(),
            },
        }
    },
    "mappings": {
        "properties": {
            **{
                f: {
                    "type": "text",
                    "analyzer": "spanish_custom",
                    "search_analyzer": SEARCH_ANALYZER,
                }
                for f in TEXT_FIELDS
            },
            **{
                f: {
                    "type": "text",
                    "analyzer": "spanish_custom",
                    "search_analyzer": SEARCH_ANALYZER,
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                }
                for f in FACET_TEXT_FIELDS
            },
            **{f: {"type": "keyword"} for f in KEYWORD_FIELDS},
            "canales_de_atencion": {"type": "keyword"},
            "tiene_costo": {"type": "boolean"},
            "otros_costos": {
                "type": "text",
                "analyzer": "spanish_custom",
                "search_analyzer": SEARCH_ANALYZER,
            },
        }
    },
}


def row_to_doc(row: pd.Series) -> dict:
    doc = {}
    for field in TEXT_FIELDS + FACET_TEXT_FIELDS + KEYWORD_FIELDS + ["otros_costos"]:
        value = row.get(field, "")
        doc[field] = value if isinstance(value, str) else ""
    # canales: "internet, persona" -> ["internet", "persona"]
    canales = row.get("canales_de_atencion", "")
    doc["canales_de_atencion"] = [
        c.strip() for c in canales.split(",") if c.strip()
    ]
    doc["tiene_costo"] = row.get("tiene_costo", "") == "1"
    return doc


def main() -> None:
    recreate = "--recreate" in sys.argv
    es = Elasticsearch(ES_URL)
    if not es.ping():
        sys.exit("No se pudo conectar a Elasticsearch en " + ES_URL)

    if recreate and es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print(f"Índice '{INDEX}' eliminado")

    if not es.indices.exists(index=INDEX):
        es.indices.create(index=INDEX, body=INDEX_SETTINGS)
        print(f"Índice '{INDEX}' creado con analyzer español")

    df = pd.read_csv(DATA_CSV, dtype=str, keep_default_na=False)
    print(f"Ingestando {len(df)} trámites...")

    t0 = time.time()
    actions = (
        {"_index": INDEX, "_id": row["id"], "_source": row_to_doc(row)}
        for _, row in df.iterrows()
    )
    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    print(f"Ingesta: {success} docs en {time.time() - t0:.1f}s, {len(errors)} errores")

    es.indices.refresh(index=INDEX)
    print(f"Total en índice: {es.count(index=INDEX)['count']}")


if __name__ == "__main__":
    main()
