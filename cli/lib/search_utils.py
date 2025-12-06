import json
import os
from pathlib import Path
from typing import Any

DOCUMENT_PREVIEW_LENGTH = 100
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 0
PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_ALPHA_VALUE = 0.5
DEFAULT_K_VALUE = 60


def load_movies():
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]


def load_stopwords():
    with open(STOPWORDS_PATH, "r") as file:
        return file.read().splitlines()


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": score,
        "metadata": metadata if metadata else {},
    }
