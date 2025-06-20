import streamlit as st
import logging
import os
import sys
import shutil
import tempfile
from typing import List
# No longer need httpx or json for direct access

# Import your core RAG components directly
# Note: Importing utils here will configure logging and initialize models/clients
import utils # This import runs code in utils.py
from doc_processing import (
    DocLoader,
    DocEmbedder,
    DocRetriever,
    VECTORDB_DIR,
    MANIFEST_PATH,
    calculate_file_hash,
    load_manifest,
    save_manifest,
    _cleanup_cache_files,
)
from doc_rag import ManagerRAG

# --- Configuration ---
DEFAULT_PDF_FOLDER_LOCAL = "pdf"

# --- Chat Memory Configuration ---
# Store the last N *turns* (user + assistant)
MAX_CHAT_TURNS = 5
# Total number of messages (user + assistant)
MAX_CHAT_MESSAGES = MAX_CHAT_TURNS * 2

# --- Basic Logging for the Streamlit App ---
# Logging is already configured in utils.py upon import
# Get the logger for the Streamlit app process
logger = logging.getLogger('StreamlitUI')

logger.info("Streamlit UI application starting up.")

# --- Ensure local PDF folder exists ---
os.makedirs(DEFAULT_PDF_FOLDER_LOCAL, exist_ok=True)
logger.info(f"Ensured local PDF folder exists at '{DEFAULT_PDF_FOLDER_LOCAL}'.")

# --- Session State Initialization ---
# Initialize session state for RAG orchestrator and chat history
if 'rag_orchestrator' not in st.session_state:
    st.session_state.rag_orchestrator = None
if 'messages' not in st.session_state:
    # Initialize chat history with a welcome message
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your RAG assistant. Please initialize the RAG system first using the sidebar."}]
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False


# --- RAG System Initialization and Ingestion ---
# This function will initialize the RAG orchestrator and run ingestion
# It needs to be triggered by a button or automatically on first load
def initialize_rag_system(pdf_folder_path: str = DEFAULT_PDF_FOLDER_LOCAL):
    logger.info("Attempting to initialize RAG system.")
    try:
        # Initialize the RAG orchestrator
        orchestrator = ManagerRAG()
        logger.info("RAG Orchestrator initialized.")

        logger.info(f"Starting document ingestion for folder '{pdf_folder_path}'.")
        # The ingestion method uses tqdm internally for console progress.
        # This progress will appear in the Streamlit server console.
        orchestrator.ingest_pdfs_from_folder(pdf_folder_path)
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
    logger.info(f"Listing local PDF files in '{pdf_folder}'")
    if not os.path.isdir(pdf_folder):
        logger.warning(f"Local PDF folder '{pdf_folder}' not found. Creating it.")
        os.makedirs(pdf_folder, exist_ok=True)
        return []
    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_folder, f))]
        logger.info(f"Found {len(pdf_files)} local PDF files.")
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing local PDF files in '{pdf_folder}': {e}", exc_info=True)
        return []

def add_uploaded_pdf_to_folder(uploaded_file, pdf_folder: str = DEFAULT_PDF_FOLDER_LOCAL) -> str:
    logger.info(f"Attempting to save uploaded PDF '{uploaded_file.name}' to local folder '{pdf_folder}'")
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
        logger.info(f"Successfully saved uploaded file '{uploaded_file.name}' to '{target_path}'.")
        return target_path
    except Exception as e:
        logger.error(f"Error saving uploaded file '{uploaded_file.name}' to '{pdf_folder}': {e}", exc_info=True)
        raise


def remove_pdf_from_folder(file_name: str, pdf_folder: str = DEFAULT_PDF_FOLDER_LOCAL):
    logger.info(f"Attempting to remove PDF '{file_name}' from local folder '{pdf_folder}'")
    target_path = os.path.join(pdf_folder, file_name)

    if not os.path.exists(target_path):
        logger.warning(f"File '{file_name}' not found in '{pdf_folder}'. Nothing to remove.")
        st.warning(f"File '{file_name}' not found in the local folder.")
        return

    try:
        os.remove(target_path)
        logger.info(f"Successfully removed '{file_name}' from '{pdf_folder}'.")
        st.success(f"Removed '{file_name}' from local folder.")
    except Exception as e:
        logger.error(f"Error removing file '{file_name}' from '{pdf_folder}': {e}", exc_info=True)
        st.error(f"Error removing file '{file_name}': {e}")
        raise


# --- Streamlit App UI Layout ---

st.title("Streamlit RAG PDF Manager & Chat")

# Sidebar for management controls
st.sidebar.header("RAG System Management")

# Button to Initialize/Re-initialize RAG
if st.sidebar.button("Initialize RAG & Ingest Documents"):
    with st.spinner("Initializing RAG and ingesting documents..."):
        initialize_rag_system(DEFAULT_PDF_FOLDER_LOCAL)
    st.rerun()


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
                    target_path = add_uploaded_pdf_to_folder(uploaded_file, DEFAULT_PDF_FOLDER_LOCAL)
                st.success(f"Saved '{uploaded_file.name}' to '{DEFAULT_PDF_FOLDER_LOCAL}'.")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred while saving the file: {e}")
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

# Apply CSS for ample lines in chat input and potentially scrollable messages
st.markdown(
    """
    <style>
    /* Style for the chat input text area */
    textarea[data-testid="stChatInputTextArea"] {
        height: 100px; /* Adjust height as needed for more lines */
    }

    /* Optional: Style for message content to be scrollable if very long */
    .stChatMessage {
        max-height: 400px; /* Max height before scrolling */
        overflow-y: auto;
    }
    .stChatMessage > div:first-child {
        max-height: 380px; /* Adjust based on padding */
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

    # Get the last few messages for history, INCLUDING the current user message
    # We will pass history from the start of the list up to and including the current user message.
    # The RAG components (QA/Ranker) are responsible for deciding how to use this history
    # and might slice it further (e.g., exclude the last message from history for prompt formatting).
    start_index_for_history = max(0, len(st.session_state.messages) - MAX_CHAT_MESSAGES)
    history_for_rag = st.session_state.messages[start_index_for_history:]
    logger.debug(f"Passing {len(history_for_rag)} messages to RAG query method for context.")


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
                # ManagerRAG.query signature needs to accept this history parameter
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

    # The history is now managed by slicing *before* passing it to the RAG query.
    # The full history is kept in st.session_state.messages, including the new turn.
    # If len(st.session_state.messages) > MAX_CHAT_MESSAGES, slicing for the *next* turn will handle truncation.
    # No need to explicitly truncate the main st.session_state.messages list here.


logger.info("Streamlit UI application finished execution cycle.")