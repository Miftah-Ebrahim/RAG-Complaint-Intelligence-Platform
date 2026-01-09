import unittest
from unittest.mock import MagicMock, patch
from src.rag import get_rag_chain


class TestRAG(unittest.TestCase):
    @patch("src.rag.Chroma")
    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.HuggingFaceAPIWrapper")
    def test_rag_chain_initialization(self, mock_llm, mock_embed, mock_chroma):
        # Mock VectorDB and Retriever
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_chroma.return_value = mock_db
        mock_db.as_retriever.return_value = mock_retriever

        # Call function
        chain = get_rag_chain()

        # Assertions
        mock_chroma.assert_called()
        mock_db.as_retriever.assert_called_with(search_kwargs={"k": 3})
        self.assertIsNotNone(chain)


if __name__ == "__main__":
    unittest.main()
