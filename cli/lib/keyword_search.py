from operator import truediv
import pickle
import os
import string
from collections import defaultdict
from nltk.stem import PorterStemmer
from .search_utils import CACHE_DIR, load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f'{movie["title"]} {movie["description"]}'
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    print(f"Searching for: {query}")
    movies = load_movies()

    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_tokens(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    final_tokens = []
    for token in tokens:
        if not token:
            continue
        if token in stopwords:
            continue
        final_tokens.append(stemmer.stem(token))

    return final_tokens
