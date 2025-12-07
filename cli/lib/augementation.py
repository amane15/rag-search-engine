from lib.ask_gemini import ask_gemini


def summarize_response(query, results):
    results = ",".join(
        [f"{res['id']}: {res['title']} - {res['document']}" for res in results]
    )
    prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {results}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
    response = ask_gemini(prompt)
    res = (response.text() or "").strip().strip('"')
    return res


def rag_response(query, results):
    docs = ",".join(
        [f"{res['id']}: {res['title']} - {res['document']}" for res in results]
    )
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""

    response = ask_gemini(prompt)
    res = (response.text() or "").strip().strip('"')
    return res


def add_citations(query, results):
    documents = ",".join(
        [f"{res['id']}: {res['title']} - {res['document']}" for res in results]
    )

    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = ask_gemini(prompt)
    res = (response.text() or "").strip().strip('"')
    return res
