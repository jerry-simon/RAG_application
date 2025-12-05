import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder
from src.vectorstore.vectorstore import VectorStore

# ------------------------------
# Streamlit page setup
# ------------------------------
st.set_page_config(page_title="Document RAG App", layout="centered")

st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------------
# Initialize session state safely
# ------------------------------
def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state["rag_system"] = None
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = False
    if "history" not in st.session_state:
        st.session_state["history"] = []


# ------------------------------
# RAG initialization
# ------------------------------
@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.DOC_CHUNK_SIZE,
            chunk_overlap=Config.DOC_CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_url(urls)
        vector_store.create_retriever(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )

        graph_builder.build_graph()
        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"Error during RAG system initialization: {e}")
        return None, 0


# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    """Main function to run the Streamlit app."""
    st.title("Document Retrieval-Augmented Generation (RAG) App")
    st.markdown("Ask questions based on the ingested documents.")

    # ✅ Initialize session state FIRST
    init_session_state()

    # --------------------------
    # Initialize the RAG system
    # --------------------------
    if not st.session_state["initialized"]:
        with st.spinner("Initializing RAG system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state["rag_system"] = rag_system
                st.session_state["initialized"] = True
                st.success(f"RAG system initialized with {num_chunks} documents.")
            else:
                st.error("Failed to initialize RAG system.")
                return

    # --------------------------
    # Question input
    # --------------------------
    st.markdown("### Ask a Question")

    with st.form(key='question_form'):
        question = st.text_input(
            "Enter your question here:",
            placeholder="e.g., What is the main topic of the documents?"
        )
        submit_button = st.form_submit_button(label='Submit')

    # --------------------------
    # Response generation
    # --------------------------
    if submit_button and question:
        if st.session_state["rag_system"]:
            with st.spinner("Generating answer..."):
                start_time = time.time()
                result = st.session_state["rag_system"].run(question)
                elapsed_time = time.time() - start_time

                st.session_state["history"].append({
                    "question": question,
                    "answer": result["answer"],
                    "time": elapsed_time
                })

                st.markdown("### Answer")
                st.success(result["answer"])

                with st.expander("Show Retrieved Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], start=1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                            height=150,
                            disabled=True
                        )

                st.caption(f"Answer generated in {elapsed_time:.2f} seconds.")

    # --------------------------
    # Conversation History
    # --------------------------
    history = st.session_state.get("history", [])
    if history:
        st.markdown("----")
        st.markdown("### Conversation History")

        for item in reversed(history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Answered in {item['time']:.2f} seconds.")
                st.markdown("----")


# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    main()