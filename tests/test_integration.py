"""Integration test for the full RAG pipeline.

Mocks the LLM and Chroma retriever to validate that the chain
returns the expected ``{result, source_documents}`` schema without
requiring live API keys or a populated vector store.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestRAGIntegration(unittest.TestCase):
    """End-to-end test of the RAG chain with mocked externals."""

    @patch("src.rag.os.getenv", return_value="fake-token")
    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.Chroma")
    def test_pipeline_returns_expected_schema(
        self,
        mock_chroma: MagicMock,
        mock_embed: MagicMock,
        mock_getenv: MagicMock,
    ) -> None:
        """Invoke the chain and assert the output dict has required keys."""
        from src.rag import get_rag_chain

        # Arrange — fake documents the retriever will return
        fake_docs = [
            Document(
                page_content="Customer was charged twice.",
                metadata={"product": "Credit card"},
            ),
            Document(
                page_content="Late fee was not reversed.",
                metadata={"product": "Credit card"},
            ),
        ]

        # Mock retriever
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = fake_docs
        mock_chroma.return_value = mock_db
        mock_db.as_retriever.return_value = mock_retriever

        # Build the chain and then patch the LLM's _call method
        # so it returns a plain string (as the real LLM would).
        with patch("src.custom_llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {"message": {"content": "Duplicate charges are a recurring issue."}}
                ]
            }
            mock_post.return_value = mock_response

            chain = get_rag_chain()
            result = chain.invoke({"query": "What are common credit card complaints?"})

        # Assert — output must be a dict with 'result' and 'source_documents'
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)
        self.assertIn("source_documents", result)

        # source_documents should be the same docs returned by the retriever
        self.assertIsInstance(result["source_documents"], list)
        self.assertTrue(len(result["source_documents"]) > 0)

        # result should be a string
        self.assertIsInstance(result["result"], str)
        self.assertIn("Duplicate charges", result["result"])

    @patch("src.rag.os.getenv", return_value="fake-token")
    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.Chroma")
    def test_pipeline_handles_empty_retrieval(
        self,
        mock_chroma: MagicMock,
        mock_embed: MagicMock,
        mock_getenv: MagicMock,
    ) -> None:
        """Chain should not crash when the retriever returns no documents."""
        from src.rag import get_rag_chain

        # Mock retriever returning empty list
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_chroma.return_value = mock_db
        mock_db.as_retriever.return_value = mock_retriever

        # Mock the actual HTTP call inside the LLM wrapper
        with patch("src.custom_llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "No relevant data found."}}]
            }
            mock_post.return_value = mock_response

            chain = get_rag_chain()
            result = chain.invoke({"query": "Unknown topic"})

        # Assert — should still work, returning empty source_documents
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)
        self.assertIn("source_documents", result)
        self.assertEqual(len(result["source_documents"]), 0)


if __name__ == "__main__":
    unittest.main()
