import os


from lib.query_reranking import rerank
from lib.query_enhancement import enhance_query
from lib.search_utils import (
    DEFAULT_ALPHA_VALUE,
    DEFAULT_K_VALUE,
    format_search_result,
    load_movies,
)
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.evaluate_query import evaluate_response


class HybridSearch:
    def __init__(self, documents) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_result(bm25_result, semantic_result, alpha)
        return combined[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)

        return fused[:limit]


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def combine_search_result(bm25_results, semantic_results, alpha=DEFAULT_ALPHA_VALUE):
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        hybrid_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=hybrid_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores):
    if len(scores) == 0:
        return
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized_scores.append(norm_score)

    return normalized_scores


def reciprocal_rank_fusion(bm25_results, semantic_results, k: int = DEFAULT_K_VALUE):
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, 1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, 1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    rrf_results = []
    for doc_id, data in rrf_scores.items():
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)


def normalize_scores_command(scores):
    normalized_scores = normalize_scores(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA_VALUE, limit: int = 5
):
    movies = load_movies()
    searcher = HybridSearch(movies)
    result = searcher.weighted_search(query, alpha, limit)

    for i, res in enumerate(result, 1):
        print(f"{i}. {res['title']}")
        print(f"   Hybrid Score: {res.get('score', 0):.3f}")
        metadata = res.get("metadata", {})
        if "bm25_score" in metadata and "semantic_score" in metadata:
            print(
                f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
            )
        print(f"   {res['document'][:100]}...")
        print()


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K_VALUE,
    limit: int = 5,
    enhance=None,
    rerank_method=None,
    evaluate=False,
):
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_limit = limit * 3 if rerank_method else limit

    print("Original query", query)
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")

    print("Enhanced query", query)

    results = searcher.rrf_search(query, k, search_limit)

    print("After evaluating")
    if evaluate:
        evaluate_response(query, results)

    print("After rrf search")
    print(results)

    if rerank_method:
        results = rerank(query, results, limit, rerank_method)

    print("After reranking")
    print(results)
    #
    # for i, res in enumerate(results, 1):
    #     rating = 0
    #     if "rating" in res:
    #         rating = res["rating"]
    #     print(f"{i}. {res['title']} {rating}/3")
    #     if rerank_method:
    #         print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
    #     print(f"   RRF Score: {res.get('score', 0):.3f}")
    #     metadata = res.get("metadata", {})
    #     ranks = []
    #     if metadata.get("bm25_rank"):
    #         ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
    #     if metadata.get("semantic_rank"):
    #         ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
    #     if ranks:
    #         print(f"   {', '.join(ranks)}")
    #     print(f"   {res['document'][:100]}...")

    return results[:limit]
