from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from src.config import VECTOR_STORE_DIR
from src.custom_llm import HuggingFaceAPIWrapper

load_dotenv()


def get_rag_chain():
    """Returns the QA chain using a pre-built vector store, using LCEL and custom LLM."""
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = str(VECTOR_STORE_DIR)
    if not VECTOR_STORE_DIR.exists():
        print(f"Warning: Vector store at {persist_dir} does not exist.")

    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Use Custom Wrapper to bypass buggy langchain-huggingface Endpoint
    llm = HuggingFaceAPIWrapper(
        repo_id="deepseek-ai/DeepSeek-R1",
        api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.1,
    )

    template = """<s>[INST] You are the **Lead Customer Insights Analyst** at CrediTrust Financial.

**YOUR DUAL MODES OF OPERATION:**

**MODE 1: GENERAL ASSISTANCE (No Context Needed)**
* If the user asks "Who are you?", "What can you do?", or "Hello":
    * Introduce yourself as the CrediTrust AI Analyst.
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL Chain
    def input_mapper(inputs):
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

    # Output adapter to match {"result": ..., "source_documents": ...}
    def final_adapter(x):
        return {"result": x["result"], "source_documents": x["context"]}

    return full_chain | final_adapter
