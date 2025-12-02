import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.documents_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Text value should not be empty of just space")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc
            movies.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        similarities = []

        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_command():
    sm = SemanticSearch()
    print(f"Model loaded: {sm.model}")
    print(f"Max sequence length: {sm.model.max_seq_length}")


def embed_text_command(text):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings_command():
    sm = SemanticSearch()
    movies = load_movies()
    embeddings = sm.load_or_create_embeddings(movies)

    print(f"Number of docs: {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text_command(query: str):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    sm = SemanticSearch()
    movies = load_movies()
    sm.load_or_create_embeddings(movies)
    search_result = sm.search(query, limit)

    for index, result in enumerate(search_result):
        print(f"{index}. {result['title']} ({result['score']})")
        print(f"{result['description']}")
