"""CrediTrust AI — Streamlit application entry point.

Provides a chat-based interface for querying the RAG complaint
intelligence pipeline.  Users can filter by product category,
view the model's chain-of-thought reasoning, and inspect source
evidence documents.
"""

import sys
import traceback
from typing import Any, Dict, Optional, Tuple

import streamlit as st

try:
    from src.rag import get_rag_chain
    from src.utils import parse_deepseek_response
except ImportError as exc:
    st.error(f"Setup Error: {exc}")
    st.stop()

st.set_page_config(page_title="CrediTrust AI", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 CrediTrust AI")
    st.markdown("---")
    st.subheader("Filter Context")
    product: str = st.selectbox(
        "Select Product", ["All Products", "Credit Card", "Mortgage", "Student Loan"]
    )
    st.markdown("---")
    st.info("A production-ready RAG system for complaint intelligence.")
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Created by **Miftah Ebrahim** for the 10 Academy Challenge.")

st.title("💬 Complaint Intelligence Assistant")

# ---------------------------------------------------------------------------
# Initialize Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_qa_chain() -> Any:
    """Load and cache the RAG chain as a Streamlit resource.

    Returns:
        The configured RAG chain runnable, or ``None`` if initialisation
        fails (an error is displayed to the user).
    """
    try:
        return get_rag_chain()
    except Exception as exc:
        st.error(
            f"Failed to initialise the RAG chain: {exc}. "
            "Please check the vector store and API token."
        )
        return None


qa = load_qa_chain()

# ---------------------------------------------------------------------------
# Display History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# User Input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the complaints..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if qa is None:
            st.error(
                "The RAG chain is not available. Please check the setup and reload."
            )
        else:
            with st.spinner("Analyzing complaints..."):
                try:
                    # Enhance prompt with product filter if applicable
                    full_query: str = (
                        f"{prompt} (Context: {product})"
                        if product != "All Products"
                        else prompt
                    )

                    response: Dict[str, Any] = qa.invoke({"query": full_query})
                    raw_answer: str = response["result"]
                    sources = response["source_documents"]

                    # Parse DeepSeek Thinking
                    thinking: Optional[str]
                    final_answer: str
                    thinking, final_answer = parse_deepseek_response(raw_answer)

                    if thinking:
                        with st.expander("💭 View Thinking Process"):
                            st.markdown(thinking)

                    st.markdown(final_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )

                    with st.expander("🔍 View Source Evidence"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Evidence #{i + 1}**")
                            st.caption(doc.page_content[:400] + "...")
                            st.divider()

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.code(traceback.format_exc())
