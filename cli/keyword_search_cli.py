#!/usr/bin/env python3

import argparse
import json
import string

punctuation_remove_table = str.maketrans("", "", string.punctuation)


def search(query):
    print(f"Searching for: {query}")
    with open("data/movies.json", "r") as file:
        movies = json.load(file)

    with open("data/stopwords.txt", "r") as file:
        stopwords = file.read().splitlines()

    result = []
    sanitized_query = query.lower().translate(punctuation_remove_table)
    tokenized_query = sanitized_query.split(" ")
    for movie in movies["movies"]:

        sanitized_movie_title = (
            movie["title"].lower().translate(punctuation_remove_table)
        )

        for token in tokenized_query:
            if token in stopwords:
                continue
            if token in sanitized_movie_title:
                result.append(movie)

    for i, movie in enumerate(result):
        print(f'{i+1}. {movie["title"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
