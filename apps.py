import streamlit as st
import logging
import os
import sys
import shutil # To copy files
import httpx # To make HTTP requests
import json # To parse JSON responses
from typing import List, Dict, Any, Optional # For type hints
import tempfile # For handling uploaded files

# --- Configuration ---
# Default folder where PDF files are stored (relative to where the backend runs)
# Note: This needs to match the default in your backend's IngestRequest model ('pdf')
DEFAULT_PDF_FOLDER_BACKEND = "pdf"

# Default URL for the FastAPI backend (backend must be running separately)
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Basic Logging for the Streamlit App ---
# Configure logging to the console for the Streamlit app process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('StreamlitUI')

logger.info("Streamlit UI application starting up.")


# --- Session State Initialization ---
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if 'backend_status' not in st.session_state:
    st.session_state.backend_status = None # To store the last checked status


# --- Backend API Interaction Functions (Synchronous for Streamlit UI) ---

def check_backend_status(backend_url: str) -> Dict[str, Any]:
    """Checks the status endpoint of the backend API."""
    logger.info(f"Checking backend status at {backend_url}/status")
    try:
        # Use synchronous client for simplicity in Streamlit callbacks
        with httpx.Client() as client:
            # Short timeout for status check
            response = client.get(f"{backend_url}/status", timeout=10.0)
            response.raise_for_status() # Raise HTTPStatusError for 4xx or 5xx responses
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Backend connection error during status check: {e}", exc_info=True)
        return {"status": "Error", "message": f"Connection error: {e}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend returned error status {e.response.status_code} during status check: {e.response.text}", exc_info=True)
        return {"status": "Error", "message": f"HTTP error {e.response.status_code}: {e.response.text[:100]}..."}
    except Exception as e:
        logger.error(f"Unexpected error during backend status check: {e}", exc_info=True)
        return {"status": "Error", "message": f"Unexpected error: {e}"}


def trigger_ingestion(backend_url: str, pdf_folder: str = DEFAULT_PDF_FOLDER_BACKEND) -> Dict[str, Any]:
    """Triggers the document ingestion process on the backend."""
    logger.info(f"Triggering backend ingestion for folder '{pdf_folder}' at {backend_url}/ingest")
    try:
        # Use synchronous client
        with httpx.Client() as client:
            # Sending the folder path as JSON body
            # Use a long timeout for ingestion as it can be time-consuming
            response = client.post(f"{backend_url}/ingest", json={"pdf_folder_path": pdf_folder}, timeout=600.0)
            response.raise_for_status()
            return response.json() # Backend should return status/message
    except httpx.RequestError as e:
        logger.error(f"Backend ingestion request failed: {e}", exc_info=True)
        return {"status": "Error", "message": f"Request error: {e}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend returned error status {e.response.status_code} during ingestion: {e.response.text}", exc_info=True)
        return {"status": "Error", "message": f"HTTP error {e.response.status_code}: {e.response.text[:100]}..."}
    except Exception as e:
        logger.error(f"Unexpected error during backend ingestion trigger: {e}", exc_info=True)
        return {"status": "Error", "message": f"Unexpected error: {e}"}

# --- Local File Management Functions (Interact with the local PDF folder that the backend reads from) ---

def list_local_pdfs(pdf_folder: str = DEFAULT_PDF_FOLDER_BACKEND) -> List[str]:
    """Lists PDF files in the local PDF folder."""
    logger.info(f"Listing local PDF files in '{pdf_folder}'")
    if not os.path.isdir(pdf_folder):
        logger.warning(f"Local PDF folder '{pdf_folder}' not found. Creating it.")
        os.makedirs(pdf_folder, exist_ok=True) # Create the folder if it doesn't exist
        return []
    try:
        # List files ending with .pdf and ensure they are actual files
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_folder, f))]
        logger.info(f"Found {len(pdf_files)} local PDF files.")
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing local PDF files in '{pdf_folder}': {e}", exc_info=True)
        return []

def add_uploaded_pdf_to_folder(uploaded_file, pdf_folder: str = DEFAULT_PDF_FOLDER_BACKEND) -> str:
    """Saves an uploaded Streamlit file object to the local PDF folder."""
    logger.info(f"Attempting to save uploaded PDF '{uploaded_file.name}' to local folder '{pdf_folder}'")
    if not uploaded_file:
        raise ValueError("No file uploaded.")

    # Ensure the target folder exists
    os.makedirs(pdf_folder, exist_ok=True)

    filename = uploaded_file.name
    target_path = os.path.join(pdf_folder, filename)

    if os.path.exists(target_path):
         logger.warning(f"File '{filename}' already exists in '{pdf_folder}'. Overwriting.")

    try:
        # Write the uploaded file's content to the target path
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer()) # getbuffer() works with Streamlit UploadedFile
        logger.info(f"Successfully saved uploaded file '{uploaded_file.name}' to '{target_path}'.")
        return target_path # Return the path where it was saved
    except Exception as e:
        logger.error(f"Error saving uploaded file '{uploaded_file.name}' to '{pdf_folder}': {e}", exc_info=True)
        raise


def remove_pdf_from_folder(file_name: str, pdf_folder: str = DEFAULT_PDF_FOLDER_BACKEND):
    """Removes a PDF file from the local PDF folder."""
    logger.info(f"Attempting to remove PDF '{file_name}' from local folder '{pdf_folder}'")
    target_path = os.path.join(pdf_folder, file_name)

    if not os.path.exists(target_path):
        logger.warning(f"File '{file_name}' not found in '{pdf_folder}'. Nothing to remove.")
        return # File not found, consider it successful removal operation

    try:
        os.remove(target_path)
        logger.info(f"Successfully removed '{file_name}' from '{pdf_folder}'.")
    except Exception as e:
        logger.error(f"Error removing file '{file_name}' from '{pdf_folder}': {e}", exc_info=True)
        raise


# --- Streamlit App UI ---

st.title("RAG Backend PDF Manager")

# --- Sidebar for Configuration ---
st.sidebar.header("Backend Configuration")
st.session_state.backend_url = st.sidebar.text_input(
    "Backend URL",
    value=st.session_state.backend_url,
    help="URL of the running FastAPI backend (e.g., http://127.0.0.1:8000)"
)

# --- Backend Status Section ---
st.header("Backend Status")

# Button to check status
if st.button("Check Backend Status"):
    with st.spinner("Checking..."):
        status_result = check_backend_status(st.session_state.backend_url)
        st.session_state.backend_status = status_result # Update status in session state

# Display last checked status
if st.session_state.backend_status:
    status = st.session_state.backend_status
    st.subheader(f"Status: {status.get('status', 'Unknown')}")
    st.write(f"Message: {status.get('message', 'N/A')}")
    if status.get("status") == "Running":
        st.write(f"RAG Orchestrator: {status.get('rag_orchestrator_status', 'N/A')}")
        st.write(f"Retriever: {status.get('retriever_status', 'N/A')}")
        st.write(f"Loaded Sources Count: {status.get('loaded_sources_count', 0)}")
        if status.get("loaded_sources_count", 0) > 0 and status.get("loaded_sources", []):
            st.write("Loaded Sources:")
            st.write(status["loaded_sources"]) # Display list directly
        else:
             st.info("No sources currently loaded by the backend.")
    elif status.get("status") == "Error":
        st.error(f"Error connecting to backend: {status.get('message', 'Unknown error')}")
    else:
         st.warning(f"Backend in unexpected state: {status.get('status', 'Unknown')}")
else:
    st.info("Click 'Check Backend Status' to connect.")


# --- Document Management Section ---
st.header("Document Management")

# Expander for Local Files
with st.expander("Local PDF Files"):
    st.info(f"These are the files in the local '{DEFAULT_PDF_FOLDER_BACKEND}' folder that the backend reads from.")
    local_pdfs = list_local_pdfs(DEFAULT_PDF_FOLDER_BACKEND)
    if local_pdfs:
        st.write("Files found locally:")
        st.write(local_pdfs)
    else:
        st.write(f"No PDF files found in local folder '{DEFAULT_PDF_FOLDER_BACKEND}'.")

# Expander for Adding Files
with st.expander("Add PDF File"):
    uploaded_file = st.file_uploader("Upload a PDF file to add", type=["pdf"])
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}")
        st.info(f"Click 'Add PDF & Trigger Ingestion' to copy this file to the backend's folder ('{DEFAULT_PDF_FOLDER_BACKEND}') and initiate processing.")

        # Button to add and trigger ingestion
        if st.button("Add PDF & Trigger Ingestion"):
            if st.session_state.backend_status and st.session_state.backend_status.get("status") == "Running":
                 try:
                     # 1. Save the uploaded file to the local PDF folder
                     with st.spinner(f"Saving {uploaded_file.name} locally..."):
                         target_path = add_uploaded_pdf_to_folder(uploaded_file, DEFAULT_PDF_FOLDER_BACKEND)
                     st.success(f"Saved '{uploaded_file.name}' to '{target_path}'.")

                     # 2. Trigger backend ingestion for the folder
                     with st.spinner(f"Triggering backend ingestion for '{DEFAULT_PDF_FOLDER_BACKEND}'..."):
                         ingest_result = trigger_ingestion(st.session_state.backend_url, DEFAULT_PDF_FOLDER_BACKEND)

                     st.subheader("Ingestion Result:")
                     st.write(f"Status: {ingest_result.get('status', 'N/A')}")
                     st.write(f"Message: {ingest_result.get('message', 'N/A')}")

                     if ingest_result.get("status") == "Success":
                         st.success("Backend ingestion successful!")
                         # Optional: Re-check backend status after successful ingestion
                         st.session_state.backend_status = check_backend_status(st.session_state.backend_url)
                     else:
                         st.error("Backend ingestion failed.")

                 except Exception as e:
                     st.error(f"An error occurred during file upload or ingestion trigger: {e}")
                     logger.error(f"Error during add/ingestion from UI: {e}", exc_info=True)
            else:
                 st.warning("Backend is not running. Cannot trigger ingestion. Please check backend status.")
        # Clear the file uploader after processing or if not triggered? Streamlit reruns handle this.


# Expander for Removing Files
with st.expander("Remove PDF File"):
    st.info(f"Enter the exact name of a PDF file in the '{DEFAULT_PDF_FOLDER_BACKEND}' folder to remove it.")
    file_name_to_remove = st.text_input("File name to remove (e.g., document.pdf)")

    if st.button("Remove PDF"):
        if file_name_to_remove:
            try:
                # 1. Remove the file locally
                with st.spinner(f"Removing {file_name_to_remove} locally..."):
                    remove_pdf_from_folder(file_name_to_remove, DEFAULT_PDF_FOLDER_BACKEND)
                st.success(f"Attempted to remove '{file_name_to_remove}'.")
                st.warning("Remember to trigger backend ingestion after removing files so the backend updates its index.")
            except Exception as e:
                 st.error(f"An error occurred while removing the file: {e}")
                 logger.error(f"Error during file removal from UI: {e}", exc_info=True)
        else:
            st.warning("Please enter a file name to remove.")


# --- Trigger Full Ingestion ---
st.header("Manual Ingestion Trigger")
st.info(f"Trigger a full scan and ingestion cycle on the backend for all files currently in the '{DEFAULT_PDF_FOLDER_BACKEND}' folder.")

if st.button("Trigger Full Backend Ingestion"):
     if st.session_state.backend_status and st.session_state.backend_status.get("status") == "Running":
        with st.spinner(f"Triggering full backend ingestion for '{DEFAULT_PDF_FOLDER_BACKEND}'..."):
             ingest_result = trigger_ingestion(st.session_state.backend_url, DEFAULT_PDF_FOLDER_BACKEND)

        st.subheader("Ingestion Result:")
        st.write(f"Status: {ingest_result.get('status', 'N/A')}")
        st.write(f"Message: {ingest_result.get('message', 'N/A')}")

        if ingest_result.get("status") == "Success":
            st.success("Backend ingestion successful!")
            # Optional: Re-check backend status after successful ingestion
            st.session_state.backend_status = check_backend_status(st.session_state.backend_url)
        else:
            st.error("Backend ingestion failed.")
     else:
         st.warning("Backend is not running. Cannot trigger ingestion. Please check backend status.")

# --- Example Query Section (Optional) ---
# You could add a query section here that calls your /query endpoint
# For file management UI, this is less critical, but useful for testing
st.header("Query Backend (for testing)")
query_text = st.text_input("Enter your query:")
if st.button("Send Query"):
    if query_text and st.session_state.backend_status and st.session_state.backend_status.get("retriever_status") == "Ready":
        try:
            with st.spinner("Sending query..."):
#               # Assuming backend /query expects JSON {"query": "your query"}
                with httpx.Client() as client:
                    query_response = client.post(f"{st.session_state.backend_url}/query", json={"query": query_text}, timeout=300.0)
                    query_response.raise_for_status()
                    answer_data = query_response.json()
                    st.subheader("Answer:")
                    st.write(answer_data.get("answer", "No answer received."))
#                   # Display sources if backend returns them in 'sources' key
                    if answer_data.get("sources"):
                        st.write("Sources:")
                        st.write(answer_data["sources"])
#
        except Exception as e:
            st.error(f"Error sending query to backend: {e}")
            logger.error(f"Error sending query from UI: {e}", exc_info=True)
    elif not query_text:
        st.warning("Please enter a query.")
    else:
        st.warning("Backend is not ready for querying. Check status and ensure documents are loaded.")

logger.info("Streamlit UI application finished execution cycle.")