"""Unit tests for the RAG chain initialization."""

import unittest
from unittest.mock import MagicMock, patch


class TestRAG(unittest.TestCase):
    """Verify that ``get_rag_chain`` wires components correctly."""

    @patch("src.rag.HuggingFaceAPIWrapper")
    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.Chroma")
    def test_rag_chain_initialization(
        self,
        mock_chroma: MagicMock,
        mock_embed: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Chain should initialise without errors and call Chroma with k=RETRIEVER_K."""
        from src.rag import get_rag_chain
        from src.config import RETRIEVER_K

        # Mock VectorDB and Retriever
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_chroma.return_value = mock_db
        mock_db.as_retriever.return_value = mock_retriever

        # Call function
        chain = get_rag_chain()

        # Assertions
        mock_chroma.assert_called()
        mock_db.as_retriever.assert_called_with(search_kwargs={"k": RETRIEVER_K})
        self.assertIsNotNone(chain)


if __name__ == "__main__":
    unittest.main()
