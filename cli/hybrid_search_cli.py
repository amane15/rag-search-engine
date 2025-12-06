import argparse
from lib.search_utils import DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE
from lib.hybrid_search import (
    normalize_scores_command,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalize provided scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Score for normalization"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search", help="Weighted search"
    )
    weighted_search_parser.add_argument("query", help="Query for weighing")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA_VALUE,
        help="constant for controlling weighing score",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit",
    )

    rrf_search_parser = subparser.add_parser("rrf-search", help="RRF search")
    rrf_search_parser.add_argument("query", help="Query for weighing")
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K_VALUE,
        help="",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        required=False,
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument("--evaluate", action="store_true", required=False)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores_command(args.scores)

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
