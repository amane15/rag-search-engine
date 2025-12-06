import argparse
import json

from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as file:
        data = json.load(file)
        test_cases = data["test_cases"]

    for case in test_cases:
        results = rrf_search_command(case["query"], 60, limit)

        relevant_retrieved = 0
        for result in results:
            if result["title"] in case["relevant_docs"]:
                relevant_retrieved += 1

        precision = relevant_retrieved / len(results)
        recall = relevant_retrieved / len(case["relevant_docs"])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {case['query']}")
        print(f"- Precision@{limit}: {precision:.4f}")
        print(f"- Recall@{limit}: {recall:.4f}")
        print(f"- F1 Score: {f1_score:.4f}")
        print(f"- Retrieved {','.join([res['title'] for res in results])}")
        print(f"- Relevant {',' .join([title for title in case['relevant_docs']])}")


if __name__ == "__main__":
    main()
