"""Tesauro de sinónimos del dominio aplicado a tiempo de búsqueda.

Los usuarios buscan con vocabulario coloquial ("cédula", "libreta")
mientras el catálogo usa nombres oficiales ("Documento Nacional de
Identidad"). El filtro `synonym_graph` expande la consulta en el
`search_analyzer` de los campos de texto — no requiere reindexar.

Uso:
    python -m src.synonyms            # aplica el tesauro al índice vivo
"""

from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
INDEX = "tramites"
SEARCH_ANALYZER = "spanish_syn"

# Formato Solr: términos equivalentes separados por coma. Se expanden en
# query-time después de lowercase/asciifolding/stopwords y antes del
# stemmer, así el término expandido también se estematiza.
SYNONYMS = [
    # identidad y documentación civil
    "cedula, documento de identidad, documento nacional de identidad, dni",
    "credencial civica, credencial",
    "partida de nacimiento, acta de nacimiento",
    "certificado de matrimonio, partida de matrimonio, acta de matrimonio",
    "antecedentes penales, certificado de antecedentes, buena conducta",
    "certificado de domicilio, constancia de domicilio, prueba de domicilio",
    # movilidad
    "libreta de conducir, licencia de conducir, licencia, libreta",
    "vtv, verificacion tecnica vehicular, inspeccion vehicular",
    "patente, matricula",
    # documentación internacional
    "apostilla, apostillado, legalizacion de documentos, legalizar",
    "visa, visado",
    # trabajo y social
    "carne de salud, carnet de salud, libreta de salud",
    "jubilacion, pension, pasividad",
    "beca, becas, fondo de solidaridad",
    "desempleo, seguro de paro, subsidio por desempleo",
    # mascotas y otros frecuentes
    "perro, gato, mascota",
    "empresa, emprendimiento, unipersonal",
]


def syn_filter_def() -> dict:
    return {
        "type": "synonym_graph",
        "synonyms": SYNONYMS,
    }


def apply_to_index(es: Elasticsearch) -> None:
    """Agrega el analyzer con sinónimos al índice existente (sin reindexar).

    El índice se cierra brevemente para actualizar `settings.analysis`;
    los datos (incluidos los embeddings) se preservan.
    """
    from src.es_index import TEXT_FIELDS, FACET_TEXT_FIELDS

    # Campos de texto que usan spanish_custom: los mismos de es_index.
    text_fields = TEXT_FIELDS + FACET_TEXT_FIELDS + ["otros_costos"]

    es.indices.close(index=INDEX)
    try:
        es.indices.put_settings(index=INDEX, body={
            "analysis": {
                "filter": {"domain_synonyms": syn_filter_def()},
                "analyzer": {
                    SEARCH_ANALYZER: {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            # domain_synonyms antes del stop: las reglas
                            # contienen "de", "por", etc. y deben parsearse
                            # con los tokens aún presentes.
                            "domain_synonyms",
                            "spanish_stop",
                            "spanish_stemmer",
                        ],
                    }
                },
            }
        })
        es.indices.put_mapping(index=INDEX, body={
            "properties": {
                f: {
                    "type": "text",
                    "analyzer": "spanish_custom",
                    "search_analyzer": SEARCH_ANALYZER,
                }
                for f in text_fields
            }
        })
    finally:
        es.indices.open(index=INDEX)

    print(f"Tesauro aplicado: {len(SYNONYMS)} grupos de sinónimos, "
          f"search_analyzer '{SEARCH_ANALYZER}' en {len(text_fields)} campos")


def main() -> None:
    es = Elasticsearch(ES_URL)
    if not es.ping():
        raise SystemExit("No se pudo conectar a Elasticsearch en " + ES_URL)
    apply_to_index(es)


if __name__ == "__main__":
    main()
