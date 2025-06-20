import streamlit as st
import logging
import os
import sys
import shutil
import tempfile
from typing import List, Dict # Keep Dict for message structure

# Import RAG components and utilities
import utils # Imports utilities and configures logging
from doc_processing import (
    DocLoader, # Needed for local file operations
    DocEmbedder, # Used by ManagerRAG
    DocRetriever, # Used by ManagerRAG
    VECTORDB_DIR, # Persistence constants
    MANIFEST_PATH,
    calculate_file_hash, # Persistence helpers
    load_manifest,
    save_manifest,
    _cleanup_cache_files,
)
from doc_rag import ManagerRAG # The main orchestrator

# --- Configuration ---
DEFAULT_PDF_FOLDER_LOCAL = "pdf"

# --- Chat Memory Configuration ---
MAX_CHAT_TURNS = 5
MAX_CHAT_MESSAGES = MAX_CHAT_TURNS * 2 # Max number of messages to keep

# --- Logging ---
logger = logging.getLogger('StreamlitUI')

# --- Ensure local PDF folder exists ---
os.makedirs(DEFAULT_PDF_FOLDER_LOCAL, exist_ok=True)

# --- Session State Initialization ---
if 'rag_orchestrator' not in st.session_state:
    st.session_state.rag_orchestrator = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your RAG assistant. Please initialize the RAG system first using the sidebar."}]
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False


# --- RAG System Initialization and Ingestion ---
def initialize_rag_system(pdf_folder_path: str = DEFAULT_PDF_FOLDER_LOCAL):
    """Initializes RAG and ingests documents from folder."""
    logger.info("Attempting to initialize RAG system.")
    try:
        orchestrator = ManagerRAG()
        logger.info("RAG Orchestrator initialized.")

        logger.info(f"Starting document ingestion for folder '{pdf_folder_path}'.")
        orchestrator.ingest_pdfs_from_folder(pdf_folder_path) # tqdm progress in console
        logger.info("Document ingestion completed.")

        st.session_state.rag_orchestrator = orchestrator
        st.session_state.rag_initialized = True
        st.success("RAG system initialized and documents ingested!")

        if orchestrator.loaded_sources:
            st.subheader("Loaded Document Sources:")
            source_list = "\n".join([f"- {os.path.basename(p)}" for p in orchestrator.loaded_sources.keys()])
            st.markdown(source_list)
        else:
            st.warning("No document sources were successfully loaded from the folder.")

    except Exception as e:
        logger.critical(f"Failed to initialize RAG system or ingest documents: {e}", exc_info=True)
        st.error(f"Failed to initialize RAG system or ingest documents: {e}")
        st.session_state.rag_orchestrator = None
        st.session_state.rag_initialized = False


# --- Local File Management Functions ---

def list_local_pdfs(pdf_folder: str = DEFAULT_PDF_FOLDER_LOCAL) -> List[str]:
    """Lists PDF files in the local PDF folder."""
    if not os.path.isdir(pdf_folder):
        os.makedirs(pdf_folder, exist_ok=True)
        return []
    try:
        return [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_folder, f))]
    except Exception as e:
        logger.error(f"Error listing local PDF files in '{pdf_folder}': {e}", exc_info=True)
        return []

def add_uploaded_pdf_to_folder(uploaded_file, pdf_folder: str = DEFAULT_PDF_FOLDER_LOCAL) -> str:
    """Saves an uploaded Streamlit file object to the local PDF folder."""
    if not uploaded_file:
        raise ValueError("No file uploaded.")

    os.makedirs(pdf_folder, exist_ok=True)

    filename = uploaded_file.name
    target_path = os.path.join(pdf_folder, filename)

    if os.path.exists(target_path):
         logger.warning(f"File '{filename}' already exists in '{pdf_folder}'. Overwriting.")

    try:
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved '{uploaded_file.name}' to '{target_path}'.")
        return target_path
    except Exception as e:
        logger.error(f"Error saving uploaded file '{uploaded_file.name}' to '{pdf_folder}': {e}", exc_info=True)
        st.error(f"Error saving uploaded file: {e}")
        raise


def remove_pdf_from_folder(file_name: str, pdf_folder: str = DEFAULT_PDF_FOLDER_LOCAL):
    """Removes a PDF file from the local PDF folder."""
    target_path = os.path.join(pdf_folder, file_name)

    if not os.path.exists(target_path):
        st.warning(f"File '{file_name}' not found in the local folder.")
        return

    try:
        os.remove(target_path)
        st.success(f"Removed '{file_name}' from local folder.")
    except Exception as e:
        logger.error(f"Error removing file '{file_name}' from '{pdf_folder}': {e}", exc_info=True)
        st.error(f"Error removing file '{file_name}': {e}")
        raise


# --- Streamlit App UI Layout ---

st.title("Streamlit RAG PDF Manager & Chat")

# Sidebar for management controls
st.sidebar.header("RAG System Management")

if st.sidebar.button("Initialize RAG & Ingest Documents"):
    with st.spinner("Initializing RAG and ingesting documents..."):
        initialize_rag_system(DEFAULT_PDF_FOLDER_LOCAL)
    st.rerun() # Rerun to refresh UI based on new state

# --- Document Management Section (in sidebar) ---
st.sidebar.header("Manage Local PDF Files")
st.sidebar.info(f"Files added/removed here affect the '{DEFAULT_PDF_FOLDER_LOCAL}' folder. Click 'Initialize RAG & Ingest Documents' to update the index.")

# Expander for Local Files List
with st.sidebar.expander("View Local Files"):
    local_pdfs = list_local_pdfs(DEFAULT_PDF_FOLDER_LOCAL)
    if local_pdfs:
        st.write("Files in folder:")
        for pdf in local_pdfs:
            st.markdown(f"- {pdf}")
    else:
        st.write(f"No PDF files found in '{DEFAULT_PDF_FOLDER_LOCAL}'.")

# Expander for Adding Files
with st.sidebar.expander("Add New PDF"):
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="add_pdf_uploader")
    if uploaded_file:
        st.write(f"Selected: {uploaded_file.name}")
        if st.button("Copy PDF to Folder"):
            try:
                with st.spinner(f"Saving {uploaded_file.name} locally..."):
                    add_uploaded_pdf_to_folder(uploaded_file, DEFAULT_PDF_FOLDER_LOCAL)
                st.rerun()
            except Exception as e:
                logger.error(f"Error saving uploaded file: {e}", exc_info=True)

# Expander for Removing Files
with st.sidebar.expander("Remove Existing PDF"):
    local_pdfs_for_removal = list_local_pdfs(DEFAULT_PDF_FOLDER_LOCAL)
    if local_pdfs_for_removal:
        file_name_to_remove = st.selectbox(
            "Select file to remove",
            [""] + local_pdfs_for_removal,
            key="remove_pdf_selector"
        )
        if st.button("Remove Selected PDF"):
            if file_name_to_remove:
                try:
                    with st.spinner(f"Removing {file_name_to_remove} locally..."):
                        remove_pdf_from_folder(file_name_to_remove, DEFAULT_PDF_FOLDER_LOCAL)
                    st.rerun()
                except Exception as e:
                    pass
            else:
                st.warning("Please select a file to remove.")
    else:
        st.write("No PDF files found to remove.")


# --- Main Chat Interface ---
st.header("Chat with Documents")

# Apply CSS for chat input and message appearance
st.markdown(
    """
    <style>
    textarea[data-testid="stChatInputTextArea"] {
        height: 100px;
    }
    .stChatMessage {
        max-height: 400px;
        overflow-y: auto;
    }
    .stChatMessage > div:first-child {
        max-height: 380px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input for user query
prompt = st.chat_input("Ask about the documents...", key="chat_input", max_chars=None)


# --- Process User Query ---
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    start_index_for_history = max(0, len(st.session_state.messages) - MAX_CHAT_MESSAGES)
    history_for_rag = st.session_state.messages[start_index_for_history:]

    # Get the orchestrator from session state
    orchestrator: ManagerRAG = st.session_state.rag_orchestrator

    if orchestrator is None or not st.session_state.rag_initialized:
        logger.warning("RAG orchestrator not initialized in session. Skipping query.")
        error_message = "⚠️ RAG system is not initialized. Please click 'Initialize RAG & Ingest Documents' in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)
    elif orchestrator.retriever is None or not orchestrator.loaded_sources:
        logger.warning("RAG retriever not ready or no sources loaded. Skipping RAG query.")
        error_message = "⚠️ Documents were not loaded successfully during initialization. RAG queries cannot be processed based on documents."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
             st.markdown(error_message)
    else:
        # Process the query using the RAG orchestrator
        with st.spinner("Processing your query..."):
            try:
                # Pass the current prompt and the sliced history to the query method
                answer = orchestrator.query(prompt, history_for_rag)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    st.markdown(answer)

            except Exception as e:
                logger.error(f"An error occurred during query processing: {e}", exc_info=True)
                error_message = f"❌ An error occurred during query processing: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)