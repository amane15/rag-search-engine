import argparse


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding")
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file",
    )

    image_search_parser = subparser.add_parser("image_search")
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file",
    )

    args = parser.parse_args()

    # TODO:
    match args.command:
        case "verify_image_embedding":
            pass
        case "image_search":
            pass


if __name__ == "__main__":
    main()
