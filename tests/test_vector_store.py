import tempfile
import unittest
from pathlib import Path

from vector_store import build_vector_db, create_book_documents, get_chroma_directory


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


class CreateBookDocumentsTests(unittest.TestCase):
    def test_keeps_embedded_newlines_with_the_preceding_book(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "descriptions.txt"
            source.write_text(
                "9780000000001 First description\ncontinued text\n9780000000002 Second description\n",
                encoding="utf-8",
            )

            documents = create_book_documents(source)

        self.assertEqual([document.metadata["isbn13"] for document in documents], [9780000000001, 9780000000002])
        self.assertEqual(documents[0].page_content, "9780000000001 First description\ncontinued text")

    def test_built_database_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tagged_description.txt").write_text(
                "9780000000001 First description\n9780000000002 Second description\n", encoding="utf-8"
            )
            db = build_vector_db(root, embeddings=FakeEmbeddings())

            self.assertEqual(db._collection.count(), 2)
            self.assertEqual(len(db.similarity_search("first", k=1)), 1)
            db._client.close()

    def test_uses_an_ascii_fallback_for_a_unicode_project_path(self):
        directory = get_chroma_directory(Path("D:/语义图书推荐"))

        self.assertTrue(str(directory).isascii())


if __name__ == "__main__":
    unittest.main()
