import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


MODEL_ID = "BAAI/bge-small-en-v1.5"


def get_embedding_model_config(project_root: Path) -> tuple[str, dict]:
    env_model_path = os.getenv("BOOK_EMBEDDING_MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path).expanduser()
        if model_path.exists():
            return str(model_path), {"local_files_only": True}
        raise FileNotFoundError(
            "BOOK_EMBEDDING_MODEL_PATH 指向的目录不存在，请检查 .env 中的路径。"
        )

    local_models = [
        project_root / "models" / "bge-small-en-v1.5",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5",
    ]
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        local_models.append(Path(hf_home).expanduser() / "hub" / "models--BAAI--bge-small-en-v1.5")

    for model_dir in local_models:
        snapshots_dir = model_dir / "snapshots"
        if model_dir.is_dir() and (model_dir / "modules.json").exists():
            return str(model_dir), {"local_files_only": True}
        if snapshots_dir.is_dir():
            snapshots = sorted(
                snapshots_dir.iterdir(),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for snapshot in snapshots:
                if (snapshot / "modules.json").exists():
                    return str(snapshot), {"local_files_only": True}

    return MODEL_ID, {}


def build_vector_db(project_root: Path, rebuild: bool) -> None:
    source_file = project_root / "tagged_description.txt"
    output_dir = project_root / "chroma_db"

    if not source_file.exists():
        raise FileNotFoundError(f"Missing source file: {source_file}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not rebuild:
            print("chroma_db already exists. Use --rebuild to recreate it.")
            return
        shutil.rmtree(output_dir)

    model_name, model_kwargs = get_embedding_model_config(project_root)
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    raw_documents = TextLoader(str(source_file), encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(output_dir),
    )
    if hasattr(db, "persist"):
        db.persist()

    print(f"Vector database built at {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma vector database for the book recommender.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate an existing chroma_db directory.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    build_vector_db(Path(__file__).resolve().parents[1], args.rebuild)
