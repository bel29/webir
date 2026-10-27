"""CLI de búsqueda de trámites.

Uso:
    python -m src.search "sacar pasaporte"
    python -m src.search          # modo interactivo
"""

import sys
import textwrap
import time

from src.index import load_index
from src.preprocess import tokenize

TOP_K = 10


def search(index: dict, query: str, top_k: int = TOP_K) -> list[dict]:
    """Consulta -> lista de resultados rankeados por BM25."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = index["bm25"].get_scores(q_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in ranked[:top_k]:
        if score <= 0:
            break
        doc = dict(index["docs"][doc_id])
        doc["score"] = round(float(score), 2)
        results.append(doc)
    return results


def format_result(rank: int, doc: dict) -> str:
    lines = [f"[{rank}] {doc['nombre_tramite']}  (score: {doc['score']})"]
    lines.append(
        f"    Organismo: {doc['institucion_nombre']}"
        f" — {doc['institucion_padre_organizacional_nombre']}"
    )
    if doc.get("ques_es"):
        snippet = textwrap.shorten(doc["ques_es"], width=220, placeholder="…")
        lines.append(f"    {snippet}")
    if doc.get("tiene_costo") == "1":
        costo = f"{doc.get('valor', '')} {doc.get('unidad', '')}".strip()
        extra = textwrap.shorten(doc.get("otros_costos", ""), 120, placeholder="…")
        lines.append(f"    Costo: {costo or 'ver detalle'}  {extra}")
    else:
        lines.append("    Costo: gratuito / no informado")
    if doc.get("canales_de_atencion"):
        lines.append(f"    Canales: {doc['canales_de_atencion']}")
    lines.append(f"    URL: {doc['url']}")
    return "\n".join(lines)


def run_query(index: dict, query: str) -> None:
    t0 = time.time()
    results = search(index, query)
    elapsed = (time.time() - t0) * 1000
    print(f'\nConsulta: "{query}" — {len(results)} resultados ({elapsed:.0f} ms)\n')
    if not results:
        print("Sin resultados. Probá con otros términos.\n")
        return
    for i, doc in enumerate(results, 1):
        print(format_result(i, doc))
        print()


def main() -> None:
    index = load_index()
    if len(sys.argv) > 1:
        run_query(index, " ".join(sys.argv[1:]))
        return
    print("Buscador de trámites de Uruguay (BM25). Escribí 'salir' para terminar.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"salir", "exit", "q"}:
            break
        if query:
            run_query(index, query)


if __name__ == "__main__":
    main()
