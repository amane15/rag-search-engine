#!/usr/bin/env python

import argparse

from lib.semantic_search import (
    embed_query_text_command,
    embed_text_command,
    search_command,
    verify_command,
    verify_embeddings_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", help="Text that should be embedded")

    verify_embeddings = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings"
    )

    query_embedding = subparsers.add_parser("embedquery", help="Embed provided query")
    query_embedding.add_argument("query", help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", help="Query to be searched")
    search_parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit number of searches",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()

        case "embed_text":
            embed_text_command(args.text)

        case "verify_embeddings":
            verify_embeddings_command()

        case "embedquery":
            embed_query_text_command(args.query)

        case "search":
            search_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
