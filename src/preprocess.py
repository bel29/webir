"""Preprocesamiento de texto en español para el sistema de RI.

Pipeline: lowercase -> quitar tildes -> tokenizar -> stopwords -> stemming.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_stemmer = None
_stopwords = None


def _init():
    global _stemmer, _stopwords
    if _stemmer is None:
        try:
            _stopwords = set(stopwords.words("spanish"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords = set(stopwords.words("spanish"))
        _stemmer = SnowballStemmer("spanish")


def normalize(text: str) -> str:
    """Minúsculas y sin tildes/diacríticos."""
    if not text:
        return ""
    return unidecode(text.lower())


def tokenize(text: str) -> list[str]:
    """Texto -> lista de términos (stems) listos para indexar."""
    _init()
    tokens = _TOKEN_RE.findall(normalize(text))
    return [
        _stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 1
    ]
