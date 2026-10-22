from pathlib import Path
import argparse

from rag.vectorstore import ingest_directory, reset_domain_index


def main():
    parser = argparse.ArgumentParser(description="Ingest real policy documents into Chroma")
    parser.add_argument("--source", type=Path, default=Path("data/policies"))
    parser.add_argument("--reset-domain", choices=("academic", "corporate"),
                        help="현재 폴더 문서로 해당 분야 검색 색인을 다시 만듭니다")
    args = parser.parse_args()
    if args.reset_domain:
        print(reset_domain_index(args.reset_domain, args.source))
    else:
        print(ingest_directory(args.source))


if __name__ == "__main__":
    main()
