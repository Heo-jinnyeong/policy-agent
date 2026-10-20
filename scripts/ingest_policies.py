from pathlib import Path
import argparse

from rag.vectorstore import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Ingest real policy documents into Chroma")
    parser.add_argument("--source", type=Path, default=Path("data/policies"))
    args = parser.parse_args()
    print(ingest_directory(args.source))


if __name__ == "__main__":
    main()
