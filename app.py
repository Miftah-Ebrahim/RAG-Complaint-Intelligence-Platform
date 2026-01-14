import streamlit as st
import sys
import traceback
import os

# Add src to path just in case, though usually not needed if run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from src.rag import get_rag_chain
    from src.utils import parse_deepseek_response
except ImportError as e:
    st.error(f"Setup Error: {e}")
    st.stop()

st.set_page_config(page_title="CrediTrust AI", page_icon="🏦", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🏦 CrediTrust AI")
    st.markdown("---")
    st.subheader("Filter Context")
    product = st.selectbox(
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

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_qa_chain():
    return get_rag_chain()


qa = load_qa_chain()

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about the complaints..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            try:
                # Enhance prompt with product filter if applicable (simple logic for now)
                full_query = (
                    f"{prompt} (Context: {product})"
                    if product != "All Products"
                    else prompt
                )

                response = qa.invoke({"query": full_query})
                raw_answer = response["result"]
                sources = response["source_documents"]

                # Parse DeepSeek Thinking
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

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
