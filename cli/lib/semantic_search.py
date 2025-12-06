import json
import re
import os
from huggingface_hub.inference._generated.types import sentence_similarity
import numpy as np
from numpy._core.multiarray import format_longfloat
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import similarity
from .search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_OVERLAP,
    format_search_result,
    load_movies,
    DOCUMENT_PREVIEW_LENGTH,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        all_chunks = []
        metadata = []

        for doc_index, doc in enumerate(self.documents):
            if doc["description"] == "":
                continue
            chunks = semantic_chunking(doc["description"], 4, 1)
            all_chunks += chunks
            for chunk_index, chunk in enumerate(chunks):
                metadata.append(
                    {
                        "movie_idx": doc_index,
                        "chunk_idx": chunk_index,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as file:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
        if os.path.exists(self.chunk_metadata_path):
            with open(self.chunk_metadata_path, "r") as file:
                self.chunk_metadata = json.load(file)
        else:
            self.build_chunk_embeddings(documents)

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embeddings = self.generate_embedding(query)
        chunk_scores = []
        for index, chunk_emb in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embeddings, chunk_emb)
            chunk_scores.append(
                {
                    "chunk_idx": index,
                    "movie_idx": self.chunk_metadata["chunks"][index]["movie_idx"],
                    "score": similarity_score,
                }
            )

        movie_scores = {}
        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk["score"] > movie_scores[movie_idx]
            ):
                movie_scores[chunk["movie_idx"]] = chunk["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda m: m[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
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


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0

    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    text = text.strip()
    if text == "":
        return [""]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not sentences[0].endswith((".", "?", "!")):
        sentences = [text]

    chunks = []

    n_sentences = len(sentences)
    i = 0
    while i < n_sentences:
        chunk_sentence = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentence) <= overlap:
            break
        chunk = " ".join(chunk_sentence).strip()
        if chunk == "":
            continue
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks


def semantic_chunk_command(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):

    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks_command():
    movies = load_movies()
    chsm = ChunkedSemanticSearch()
    embeddings = chsm.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int = 5):
    chsm = ChunkedSemanticSearch()
    movies = load_movies()
    chsm.load_or_create_chunk_embeddings(movies)
    results = chsm.search_chunks(query)

    for i, result in enumerate(results):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
