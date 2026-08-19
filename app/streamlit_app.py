"""
Streamlit UI for RAG Document Chat

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable torch compilation for Windows compatibility
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_SKIP_AUTOGRAD_FALLBACK"] = "1"

# Initialize session state
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
    st.session_state.document_count = 0
    st.session_state.chunk_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def configure_torch():
    """Apply the Windows compatibility setting before loading models."""
    import torch

    torch.compile = lambda x: x


def initialize_services():
    """Initialize the underlying RAG services used by this app."""
    configure_torch()

    from app.services.chunking_service import ChunkingService
    from app.services.document_converter import HybridDocumentConverter
    from app.services.embedding import EmbeddingService
    from app.services.retrieval_service_instrumented import RetrievalService

    return {
        "converter": HybridDocumentConverter(),
        "chunker": ChunkingService(chunk_size=800, chunk_overlap=100),
        "embedding_service": EmbeddingService(),
        "retrieval_service": RetrievalService(),
    }


@st.cache_resource
def get_rag_service():
    """Keep transformer models loaded across Streamlit reruns."""
    configure_torch()

    from app.services.rag_service_instrumented import RAGService

    return RAGService()


def process_documents(uploaded_files):
    """Process uploaded PDFs and store embeddings in the vector database."""
    from app.services.vector_store import create_documents_table, insert_document

    services = initialize_services()

    st.session_state.document_count = 0
    st.session_state.chunk_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    create_documents_table()

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            status_text.text(
                f"[{idx + 1}/{len(uploaded_files)}] Processing: {uploaded_file.name}"
            )

            result = services["converter"].convert(Path(tmp_path))
            chunks = services["chunker"].chunk_docling_document(
                result.docling_document,
                document_name=uploaded_file.name,
            )

            for chunk in chunks:
                embedding = services["embedding_service"].embed(chunk.content)
                insert_document(
                    content=chunk.content,
                    embedding=embedding,
                    metadata={
                        **chunk.metadata,
                        "source": uploaded_file.name,
                    },
                )

            st.session_state.document_count += 1
            st.session_state.chunk_count += len(chunks)

            os.unlink(tmp_path)
            progress_bar.progress((idx + 1) / len(uploaded_files))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return False

    status_text.text("✓ Documents processed successfully!")
    st.session_state.documents_loaded = True
    return True


def answer_question(question: str):
    """Generate the final answer from the document corpus."""
    try:
        rag_service = get_rag_service()
        return rag_service.answer_with_sources(question)

    except Exception as e:
        return f"Error generating answer: {str(e)}", []


# Streamlit UI
st.set_page_config(page_title="RAG Document Chat", layout="wide")

# Title and description
st.title("📚 RAG Document Chat")
st.markdown(
    "Upload your documents and chat with them using AI-powered retrieval and generation."
)

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    if st.button("🔄 Clear Database", key="clear_db"):
        from app.services.vector_store import create_documents_table, drop_documents_table

        drop_documents_table()
        create_documents_table()
        st.session_state.documents_loaded = False
        st.session_state.document_count = 0
        st.session_state.chunk_count = 0
        st.success("Database cleared!")
    
    st.divider()
    
    # Upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files and st.button("Process Documents", key="process_btn"):
        with st.spinner("Processing documents..."):
            if process_documents(uploaded_files):
                st.balloons()
    
    st.divider()
    
    # Status section
    st.subheader("📊 Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.document_count)
    with col2:
        st.metric("Chunks", st.session_state.chunk_count)
    
    if st.session_state.documents_loaded:
        st.success("✓ Ready to chat!")
    else:
        st.info("💡 Upload documents to get started")


# Main content area
if not st.session_state.documents_loaded:
    st.info("👈 **Step 1:** Upload PDF documents using the sidebar")
    st.info("👈 **Step 2:** Click 'Process Documents'")
    st.info("👈 **Step 3:** Ask questions about your documents")
else:
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📖 Source Chunks"):
                    for i, chunk in enumerate(message["sources"], 1):
                        content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
                        preview = (content or "")[:200]
                        st.markdown(f"**Chunk {i}:** {preview}...")
    
    # Input area
    user_question = st.chat_input(
        "Ask a question about your documents...",
        key="user_input"
    )
    
    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(user_question)
                st.markdown(answer)
        
        # Store in chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


# Footer
st.divider()
st.markdown(
    """
    **About RAG Chat:**
    - Upload PDF documents to build a knowledge base
    - Ask questions and get answers based on your documents
    - View source chunks to see where answers come from
    - Built with Streamlit + Sentence Transformers + Llama 2
    """
)
