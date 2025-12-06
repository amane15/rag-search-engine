import time
import json
from typing import Optional
from sentence_transformers import CrossEncoder

from lib.ask_gemini import ask_gemini


def rerank_result(query: str, results: list[dict], limit=5):
    scored_docs = []
    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""

        response = ask_gemini(prompt)
        score = (response.text or "").strip().strip('"')
        scored_docs.append({**doc, "individual_score": int(score)})
        time.sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def batch_result(query, results, limit=5):
    doc_list_str = json.dumps(results)
    prompt = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """

    batch_ranking_result = ask_gemini(prompt)
    result = batch_ranking_result.text.replace("`", "").replace("json", "")
    ids = json.loads(result)

    final_list = []
    for id in ids:
        for result in results:
            if result["id"] == id:
                final_list.append(result)

    for index, result in enumerate(final_list, 1):
        result["individual_score"] = index

    return final_list[:limit]


def cross_encoder_ranking(query, results, limit):
    pairs = []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for index, score in enumerate(scores):
        results[index]["cross_score"] = score

    results.sort(key=lambda x: x["cross_score"], reverse=True)
    return results[:limit]


def rerank(
    query: str, results: list[dict], limit=5, rerank: Optional[str] = None
) -> str:
    match rerank:
        case "individual":
            return rerank_result(query, results, limit)
        case "batch":
            return batch_result(query, results, limit)
        case "cross_encoder":
            return cross_encoder_ranking(query, results, limit)
        case _:
            return query
