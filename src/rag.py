"""RAG chain construction for the CrediTrust Complaint Intelligence Platform.

Builds a LangChain Expression Language (LCEL) chain that retrieves
relevant complaint documents from a Chroma vector store and generates
analyst-quality answers via the DeepSeek-R1 LLM.
"""

from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from src.config import (
    EMBEDDING_MODEL_NAME,
    LLM_REPO_ID,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    VECTOR_STORE_DIR,
)
from src.custom_llm import HuggingFaceAPIWrapper
from src.logger import logger

load_dotenv()


def get_rag_chain() -> Runnable:
    """Build and return the full RAG chain.

    The chain performs the following steps:
      1. Map the user query into the expected schema.
      2. Retrieve the top-*k* most relevant complaint documents.
      3. Format the documents and pass them through a prompt template.
      4. Generate an answer with the LLM.
      5. Adapt the output to a standard ``{result, source_documents}`` dict.

    Returns:
        Runnable: A LangChain runnable that accepts ``{"query": str}``
            and returns ``{"result": str, "source_documents": List[Document]}``.

    Raises:
        FileNotFoundError: If the vector store directory does not exist
            (logged as a warning; retrieval may still fail at invoke time).
    """
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    persist_dir: str = str(VECTOR_STORE_DIR)
    if not VECTOR_STORE_DIR.exists():
        logger.warning(
            f"Vector store at {persist_dir} does not exist. Retrieval may fail."
        )

    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = HuggingFaceAPIWrapper(
        repo_id=LLM_REPO_ID,
        api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=LLM_TEMPERATURE,
    )

    template = """\
<s>[INST] You are the **Lead Customer Insights Analyst** at CrediTrust Financial.

**YOUR DUAL MODES OF OPERATION:**

**MODE 1: GENERAL ASSISTANCE (No Context Needed)**
* If the user asks "Who are you?", "What can you do?", or "Hello":
    * Introduce yourself as your name is Miftah  and you are the Key for the companys credit assistance  and
    *you are developed by Miftah by thier internal senior AI Engineer .
    * Explain that your job is to help Product Managers identify trends in customer complaints.
    * List the 5 products you cover: Credit Cards, Personal Loans, Savings, Money Transfers.
    * **Do not** provide an Executive Summary for these questions. Just be helpful and professional.

**MODE 2: DATA ANALYSIS (Strict Context Required)**
* If the user asks a specific question about complaints, trends, or issues:
    * **GROUND TRUTH:** Use ONLY the "Complaint Context" below.
    * **EVIDENCE:** Cite your sources or quote the text.
    * **FORMAT:** Use the strict "Executive Summary" and "Key Findings" structure.
    * **NO HALLUCINATION:** If the answer isn't in the text, say: "The current dataset lacks sufficient information."

-------------------------------------------------------------------------------
**COMPLAINT CONTEXT:**
{context}
-------------------------------------------------------------------------------

**USER QUESTION:**
{question}

**RESPONSE:**
[/INST]
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs: List[Document]) -> str:
        """Concatenate document page contents into a single context string.

        Args:
            docs: List of LangChain ``Document`` objects retrieved from the
                vector store.

        Returns:
            A newline-separated string of all document contents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def input_mapper(inputs: Dict[str, Any]) -> Dict[str, str]:
        """Map the external ``query`` key to the internal ``question`` key.

        Args:
            inputs: Dictionary containing a ``query`` key.

        Returns:
            Dictionary with ``question`` key expected by downstream steps.
        """
        return {"question": inputs["query"]}

    # Chain to get context (documents)
    retrieval_step = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["question"])
    )

    # Chain to generate answer
    generation_step = (
        RunnablePassthrough.assign(
            formatted_context=lambda x: format_docs(x["context"])
        )
        | (lambda x: {"context": x["formatted_context"], "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )

    full_chain = (
        input_mapper
        | retrieval_step
        | RunnablePassthrough.assign(result=generation_step)
    )

    def final_adapter(x: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt chain output to the standard response schema.

        Args:
            x: Raw chain output containing ``result`` and ``context`` keys.

        Returns:
            Dictionary with ``result`` (str) and ``source_documents``
            (List[Document]) keys.
        """
        return {"result": x["result"], "source_documents": x["context"]}

    return full_chain | final_adapter
