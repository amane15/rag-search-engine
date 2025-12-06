from lib.ask_gemini import ask_gemini
import json


def evaluate_response(query, results):
    formatted_results = []
    for doc in results:
        formatted_results.append(f"{doc['id']}: {doc['title']} - {doc['document']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(formatted_results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""

    response = ask_gemini(prompt)
    scores = json.loads(response.text)

    for index, doc in enumerate(results):
        doc["rating"] = scores[index]

    results.sort(key=lambda x: x["rating"], reverse=True)
