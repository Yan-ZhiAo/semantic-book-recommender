import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vector_store import build_render_vector_index, build_vector_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma vector database for the book recommender.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate an existing chroma_db directory.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    if os.getenv("BOOK_EMBEDDING_BACKEND", "sentence-transformers").lower() == "fastembed":
        build_render_vector_index(PROJECT_ROOT, rebuild=args.rebuild)
    else:
        build_vector_db(PROJECT_ROOT, rebuild=args.rebuild)
