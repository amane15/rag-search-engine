import argparse


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file",
    )
    parser.add_argument("--query", type=str, help="Query for image")

    args = parser.parse_args()

    # TODO:


if __name__ == "__main__":
    main()
