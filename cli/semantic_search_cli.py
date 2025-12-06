#!/usr/bin/env python

import argparse

from lib.semantic_search import (
    embed_query_text_command,
    embed_text_command,
    search_chunked_command,
    search_command,
    semantic_chunk_command,
    verify_command,
    verify_embeddings_command,
    chunk_command,
    embed_chunks_command,
)
from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
)


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

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunks long text into smaller pieces for embedding"
    )
    chunk_parser.add_argument("text", help="Text to be chunked")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        required=False,
        default=DEFAULT_CHUNK_SIZE,
        help="chunk size",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        required=False,
        help="Include character from previous chunk with provided overlap",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Semantically chunks long text into smaller pieces for embedding",
    )
    semantic_chunk_parser.add_argument("text", help="Text to be chunked")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        required=False,
        default=DEFAULT_MAX_CHUNK_SIZE,
        help="max chunk size",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        required=False,
        help="Include character from previous chunk with provided overlap",
    )

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed chunks")

    search_chunk_parser = subparsers.add_parser("search_chunked", help="Search chunks")
    search_chunk_parser.add_argument("query", help="Query for search")
    search_chunk_parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=5,
        help="number of matches to return",
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

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
