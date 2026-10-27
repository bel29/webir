"""Construcción del índice BM25 sobre el catálogo de trámites.

Uso:
    python -m src.index            # construye y guarda data/index.pkl
"""

import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocess import tokenize

DATA_CSV = Path("data/tramites.csv")
INDEX_PKL = Path("data/index.pkl")

# Campos indexables con su peso (repetición de tokens).
FIELD_WEIGHTS = {
    "nombre_tramite": 3,
    "ques_es": 2,
    "casuistica": 1,
    "requisitos_generales": 1,
    "otros_datos_de_interes": 1,
    "institucion_nombre": 1,
    "institucion_padre_organizacional_nombre": 1,
    "categorias_de_organismo": 1,
}

# Campos que se muestran en los resultados.
DISPLAY_FIELDS = [
    "id",
    "nombre_tramite",
    "institucion_nombre",
    "institucion_padre_organizacional_nombre",
    "ques_es",
    "tiene_costo",
    "unidad",
    "valor",
    "otros_costos",
    "canales_de_atencion",
    "internet_url_del_tramite",
    "url",
]


def load_catalog(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    print(f"Catálogo cargado: {len(df)} trámites, {len(df.columns)} columnas")
    return df


def doc_tokens(row: pd.Series) -> list[str]:
    """Tokens del documento con ponderación por campo (repetición)."""
    tokens: list[str] = []
    for field, weight in FIELD_WEIGHTS.items():
        text = row.get(field, "")
        if text:
            tokens.extend(tokenize(text) * weight)
    return tokens


def build_index(df: pd.DataFrame) -> dict:
    t0 = time.time()
    corpus = [doc_tokens(row) for _, row in df.iterrows()]
    bm25 = BM25Okapi(corpus)
    elapsed = time.time() - t0
    print(f"Índice construido en {elapsed:.1f}s ({len(corpus)} documentos)")
    return {
        "bm25": bm25,
        "docs": df[DISPLAY_FIELDS].to_dict("records"),
    }


def save_index(index: dict, path: Path = INDEX_PKL) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"Índice guardado en {path}")


def load_index(path: Path = INDEX_PKL) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_CSV
    df = load_catalog(csv_path)
    index = build_index(df)
    save_index(index)
