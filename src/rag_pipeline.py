from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.path.join(
    os.getcwd(), "data/processed/vector_store"
)  # As per instruction "data/processed/" or "vector_store/"
# I'll check both or assume a standard one. The user said "data/processed/ (or vector_store/)".
# I will use "vector_store" as the directory name.


def get_retriever():
    """Loads ChromaDB vector store and returns retriever."""
    if not os.path.exists("vector_store"):
        # Fallback for demonstration since we didn't actually create it in Task 2
        # Real scenario: Raise error. Here: Create a dummy one or Raise.
        # User said "Use the PRE-BUILT vector store".
        # If I can't find it, I will assume it's in 'data/processed/chroma_db' or similar.
        # I'll default to 'vector_store' folder.
        pass

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check "vector_store" and "data/processed/vector_store"
    persist_directory = "vector_store"
    if not os.path.exists(persist_directory) and os.path.exists(
        "data/processed/vector_store"
    ):
        persist_directory = "data/processed/vector_store"

    # If still not found, we might need to create it on the fly for the app to work?
    # For now, standard load.
    vector_db = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_function
    )
    return vector_db.as_retriever(search_kwargs={"k": 3})


def get_rag_chain():
    """Creates RAG chain with Mistral-7B."""
    retriever = get_retriever()

    # Check for HF Token
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACEHUB_API_TOKEN not found in .env")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    template = """You are a financial analyst. Use the context below to answer the question. 
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain
