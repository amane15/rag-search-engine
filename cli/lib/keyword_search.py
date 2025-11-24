import math
import pickle
import os
import string
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from .search_utils import CACHE_DIR, load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_freq.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

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
        with open(self.tf_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.tf_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    seen, results = set(), []
    query_tokens = tokenize_text(query)

    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term) * idx.get_idf(term)


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
