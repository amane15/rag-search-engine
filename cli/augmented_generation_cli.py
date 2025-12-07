import argparse

from lib.augementation import summarize_response, rag_response, add_citations
from lib.hybrid_search import rrf_search_command
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="Summarise the search result"
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("--limit", required=False, type=int, help="Limit")

    citations_parser = subparsers.add_parser(
        "citations", help="Add citation to the results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", required=False, type=int, help="Limit")

    question_parser = subparsers.add_parser(
        "question", help="Add citation to the results"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument("--limit", required=False, type=int, help="Limit")

    args = parser.parse_args()

    movies = load_movies()
    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query, 60, 5)
            rag_res = rag_response(query, results[:5])

            print("Search Results:")
            for res in results:
                print(f" - {res['title']}")
            print("RAG Response:")
            print(rag_res)
        case "summarize":
            query = args.query
            results = rrf_search_command(query, 60, 5)
            llm_summary = summarize_response(query, results)

            print("Search Results:")
            for res in results:
                print(f" - {res['title']}")
            print("LLM Summary")
            print(llm_summary)
        case "citations":
            query = args.query
            results = rrf_search_command(query, 60, 5)
            llm_citations = add_citations(query, results)

            print("Search Results:")
            for res in results:
                print(f" - {res['title']}")
            print("LLM Answer")
            print(llm_citations)
        case "question":
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
