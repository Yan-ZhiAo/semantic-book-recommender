"""Shared utilities for building and validating the book vector store."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_ID = "BAAI/bge-small-en-v1.5"
VECTOR_DB_SCHEMA_VERSION = 4
MANIFEST_NAME = "book_recommender_manifest.json"


def get_embedding_model_config(project_root: Path) -> tuple[str, dict]:
    """Prefer an available local model before falling back to Hugging Face."""
    env_model_path = os.getenv("BOOK_EMBEDDING_MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path).expanduser()
        if model_path.exists():
            return str(model_path), {"local_files_only": True}
        raise FileNotFoundError("BOOK_EMBEDDING_MODEL_PATH points to a directory that does not exist.")

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
            for snapshot in sorted(snapshots_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if (snapshot / "modules.json").exists():
                    return str(snapshot), {"local_files_only": True}
    return MODEL_ID, {}


def create_embeddings(project_root: Path) -> HuggingFaceEmbeddings:
    model_name, model_kwargs = get_embedding_model_config(project_root)
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def _source_manifest(source_file: Path) -> dict:
    with source_file.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"schema_version": VECTOR_DB_SCHEMA_VERSION, "source_sha256": digest}


def _manifest_path(project_root: Path) -> Path:
    return get_chroma_directory(project_root) / MANIFEST_NAME


def get_chroma_directory(project_root: Path) -> Path:
    """Return a Chroma path that works with its Windows native index bindings."""
    env_path = os.getenv("BOOK_CHROMA_DB_PATH")
    if env_path:
        directory = Path(env_path).expanduser()
    else:
        directory = project_root / "chroma_db"
    if str(directory).isascii():
        return directory

    # Chroma 1.x's HNSW binding cannot create indexes in Unicode paths on
    # Windows. The system temp directory is usually safe; use the drive root as
    # a final ASCII fallback if it is not.
    digest = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:12]
    fallback = Path(tempfile.gettempdir()) / "semantic-book-recommender" / digest
    if str(fallback).isascii():
        return fallback
    return Path(project_root.anchor) / "semantic-book-recommender" / digest


def is_vector_db_current(project_root: Path) -> bool:
    source_file = project_root / "tagged_description.txt"
    manifest_path = _manifest_path(project_root)
    if not source_file.exists() or not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")) == _source_manifest(source_file)
    except (OSError, json.JSONDecodeError):
        return False


def create_book_documents(source_file: Path) -> list[Document]:
    """Create one Chroma document per ISBN-prefixed book description.

    Some CSV descriptions contain embedded newlines, so a record continues until
    the next line that begins with an ISBN.
    """
    documents = []
    current_isbn = None
    current_lines = []

    def add_current_document() -> None:
        if current_isbn is not None:
            documents.append(Document(page_content="\n".join(current_lines), metadata={"isbn13": current_isbn}))

    for line_number, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        isbn, separator, _ = line.partition(" ")
        if separator and isbn.isdigit():
            add_current_document()
            current_isbn = int(isbn)
            current_lines = [line]
        elif current_isbn is None and line:
            raise ValueError(f"Invalid book description at line {line_number}: expected an ISBN prefix.")
        elif current_isbn is not None:
            current_lines.append(line)
    add_current_document()
    if not documents:
        raise ValueError(f"No book descriptions found in {source_file}")
    return documents


def build_vector_db(project_root: Path, rebuild: bool = False, embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    source_file = project_root / "tagged_description.txt"
    output_dir = get_chroma_directory(project_root)
    if not source_file.exists():
        raise FileNotFoundError(f"Missing source file: {source_file}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not rebuild and is_vector_db_current(project_root):
            print("chroma_db is current.")
            return Chroma(persist_directory=str(output_dir), embedding_function=embeddings)
        shutil.rmtree(output_dir)

    embeddings = embeddings or create_embeddings(project_root)
    db = Chroma.from_documents(
        create_book_documents(source_file),
        embedding=embeddings,
        persist_directory=str(output_dir),
        collection_configuration={"hnsw": {"sync_threshold": 1}},
    )
    # Force Chroma's asynchronous compactor to materialize the HNSW files before
    # the short-lived build process closes its client.
    db.similarity_search_by_vector(embeddings.embed_query("vector store readiness check"), k=1)
    # Chroma 1.x writes the HNSW files when its persistent client closes. Without
    # this, a short-lived build script can leave an SQLite database whose vector
    # index cannot be opened by the dashboard process.
    db._client.close()
    _manifest_path(project_root).write_text(
        json.dumps(_source_manifest(source_file), ensure_ascii=False), encoding="utf-8"
    )
    print(f"Vector database built at {output_dir}")
    return Chroma(persist_directory=str(output_dir), embedding_function=embeddings)
