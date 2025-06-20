import logging
import os
import sys
import argparse
import shutil # For copying files
import httpx # For making HTTP requests to the FastAPI backend
from typing import List, Dict, Any

# --- Basic Logging for the Manager Script ---
# Configure logging specifically for this script's console output
# This is separate from the logging configured *within* the backend/utils.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('PDFManager')

# --- Configuration ---
# Default folder where PDF files are stored (relative to where you run the script)
DEFAULT_PDF_FOLDER = "pdf"

# Default URL for the FastAPI backend
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# --- Backend API Interaction Functions ---

async def check_backend_status(backend_url: str) -> Dict[str, Any]:
    """Checks the status endpoint of the backend API."""
    logger.info(f"Checking backend status at {backend_url}/status")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/status", timeout=10.0) # Short timeout for status
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


async def trigger_ingestion(backend_url: str, pdf_folder: str = DEFAULT_PDF_FOLDER) -> Dict[str, Any]:
    """Triggers the document ingestion process on the backend."""
    logger.info(f"Triggering backend ingestion for folder '{pdf_folder}' at {backend_url}/ingest")
    try:
        async with httpx.AsyncClient() as client:
            # Sending the folder path as JSON body
            response = await client.post(f"{backend_url}/ingest", json={"pdf_folder_path": pdf_folder}, timeout=600.0) # Long timeout for ingestion
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

# --- Local File Management Functions ---

def list_local_pdfs(pdf_folder: str = DEFAULT_PDF_FOLDER) -> List[str]:
    """Lists PDF files in the local PDF folder."""
    logger.info(f"Listing local PDF files in '{pdf_folder}'")
    if not os.path.isdir(pdf_folder):
        logger.warning(f"Local PDF folder '{pdf_folder}' not found.")
        return []
    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_folder, f))]
        logger.info(f"Found {len(pdf_files)} local PDF files.")
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing local PDF files in '{pdf_folder}': {e}", exc_info=True)
        return []

def add_pdf_to_folder(source_path: str, pdf_folder: str = DEFAULT_PDF_FOLDER) -> str:
    """Copies a PDF file to the local PDF folder."""
    logger.info(f"Attempting to add PDF from '{source_path}' to local folder '{pdf_folder}'")
    if not os.path.isfile(source_path):
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not source_path.lower().endswith('.pdf'):
         logger.warning(f"Source file '{source_path}' does not look like a PDF.")

    # Ensure the target folder exists
    os.makedirs(pdf_folder, exist_ok=True)

    filename = os.path.basename(source_path)
    target_path = os.path.join(pdf_folder, filename)

    if os.path.exists(target_path):
         logger.warning(f"File '{filename}' already exists in '{pdf_folder}'. Overwriting.")

    try:
        shutil.copy2(source_path, target_path) # copy2 preserves metadata
        logger.info(f"Successfully copied '{source_path}' to '{target_path}'.")
        return target_path # Return the path where it was copied
    except Exception as e:
        logger.error(f"Error copying file '{source_path}' to '{pdf_folder}': {e}", exc_info=True)
        raise


def remove_pdf_from_folder(file_name: str, pdf_folder: str = DEFAULT_PDF_FOLDER):
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


# --- Command Line Interface (CLI) ---

async def main():
    parser = argparse.ArgumentParser(description="Manage PDFs and trigger RAG backend ingestion.")

    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help=f"Base URL of the FastAPI backend (default: {DEFAULT_BACKEND_URL})")
    parser.add_argument("--pdf-folder", default=DEFAULT_PDF_FOLDER, help=f"Local folder containing PDF files (default: {DEFAULT_PDF_FOLDER})")

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list-local
    subparsers.add_parser("list-local", help="List PDF files in the local PDF folder.")

    # Command: list-backend
    subparsers.add_parser("list-backend", help="List sources loaded by the backend RAG system.")

    # Command: add
    add_parser = subparsers.add_parser("add", help="Copy a PDF file to the local folder and trigger ingestion.")
    add_parser.add_argument("source_file", help="Path to the PDF file to add.")

    # Command: remove
    remove_parser = subparsers.add_parser("remove", help="Remove a PDF file from the local folder.")
    remove_parser.add_argument("file_name", help="Name of the PDF file to remove (must be in the PDF folder).")

    # Command: ingest (trigger ingestion directly)
    subparsers.add_parser("ingest", help="Trigger a full ingestion cycle on the backend for the configured PDF folder.")


    args = parser.parse_args()

    backend_url = args.backend_url
    pdf_folder = args.pdf_folder
    command = args.command

    if not command:
        parser.print_help()
        sys.exit(1)

    if command == "list-local":
        pdfs = list_local_pdfs(pdf_folder)
        if pdfs:
            print(f"\nLocal PDF files in '{pdf_folder}':")
            for pdf in pdfs:
                print(f"- {pdf}")
        else:
            print(f"\nNo local PDF files found in '{pdf_folder}'.")

    elif command == "list-backend":
        status = await check_backend_status(backend_url)
        if status.get("status") == "Running":
            loaded_count = status.get("loaded_sources_count", 0)
            print(f"\nBackend Status: {status.get('message', 'N/A')}")
            print(f"RAG Orchestrator Status: {status.get('rag_orchestrator_status', 'N/A')}")
            print(f"Retriever Status: {status.get('retriever_status', 'N/A')}")
            print(f"Loaded Sources Count: {loaded_count}")
            if loaded_count > 0 and status.get("loaded_sources", []):
                print("Loaded Sources:")
                for source in status["loaded_sources"]:
                    print(f"- {source}")
            elif loaded_count > 0:
                 print("Loaded Sources: (List not available from backend)")
            else:
                print("No sources currently loaded by the backend.")
        else:
            print(f"\nCould not get backend status: {status.get('message', 'Unknown error')}")
            print("Please ensure the backend (fastapi_backend.py) is running.")


    elif command == "add":
        if not hasattr(args, 'source_file'):
             parser.error("the following arguments are required: source_file")

        source_file = args.source_file
        if not os.path.exists(source_file):
             logger.error(f"Source file not found: {source_file}")
             print(f"\nError: Source file not found at '{source_file}'.")
             sys.exit(1)

        try:
            # 1. Copy file to the local PDF folder
            target_path = add_pdf_to_folder(source_file, pdf_folder)
            print(f"\nCopied '{source_file}' to '{target_path}'.")

            # 2. Trigger backend ingestion for the folder
            print("Triggering backend ingestion...")
            ingest_result = await trigger_ingestion(backend_url, pdf_folder)
            print("\nBackend Ingestion Result:")
            print(f"Status: {ingest_result.get('status', 'N/A')}")
            print(f"Message: {ingest_result.get('message', 'N/A')}")
            if ingest_result.get("status") == "Success":
                print("\nSuccessfully added PDF and triggered ingestion.")
                # Optional: Check backend status again to confirm loaded sources
                status = await check_backend_status(backend_url)
                loaded_sources = status.get("loaded_sources", [])
                if os.path.basename(target_path) in loaded_sources:
                     print(f"Confirmed '{os.path.basename(target_path)}' is now loaded by the backend.")
                else:
                     print(f"Note: '{os.path.basename(target_path)}' not yet listed as loaded by backend status.")
            else:
                 print("\nFailed to trigger backend ingestion.")

        except FileNotFoundError as e:
             print(f"\nError: {e}")
             sys.exit(1)
        except Exception as e:
             print(f"\nAn error occurred during add operation: {e}")
             logger.error(f"Error during add operation: {e}", exc_info=True)
             sys.exit(1)


    elif command == "remove":
        if not hasattr(args, 'file_name'):
             parser.error("the following arguments are required: file_name")

        file_name_to_remove = args.file_name
        try:
            # 1. Remove file from the local PDF folder
            remove_pdf_from_folder(file_name_to_remove, pdf_folder)
            print(f"\nAttempted to remove '{file_name_to_remove}' from '{pdf_folder}'.")

            # 2. Note on Backend Cleanup
            print("\nNote on Backend Cleanup:")
            print("Removing the file locally does NOT automatically remove its vector data from the backend.")
            print("The backend's data for this file will be cleaned up during the NEXT full ingestion cycle")
            print(f"when '{file_name_to_remove}' is detected as missing from the '{pdf_folder}' folder.")
            print("To ensure cleanup, run the 'ingest' command:")
            print(f"  python pdf_manager.py --backend-url {backend_url} --pdf-folder {pdf_folder} ingest")

        except Exception as e:
             print(f"\nAn error occurred during remove operation: {e}")
             logger.error(f"Error during remove operation: {e}", exc_info=True)
             sys.exit(1)


    elif command == "ingest":
         try:
            ingest_result = await trigger_ingestion(backend_url, pdf_folder)
            print("\nBackend Ingestion Result:")
            print(f"Status: {ingest_result.get('status', 'N/A')}")
            print(f"Message: {ingest_result.get('message', 'N/A')}")
            if ingest_result.get("status") == "Success":
                 print("\nSuccessfully triggered backend ingestion.")
                 # Optional: Check backend status again to confirm loaded sources
                 status = await check_backend_status(backend_url)
                 loaded_count = status.get("loaded_sources_count", 0)
                 print(f"\nBackend now reports {loaded_count} loaded source(s).")
                 if loaded_count > 0 and status.get("loaded_sources", []):
                      print("Current Loaded Sources:")
                      for source in status["loaded_sources"]:
                           print(f"- {source}")

            else:
                 print("\nFailed to trigger backend ingestion.")

         except Exception as e:
             print(f"\nAn error occurred during ingestion command: {e}")
             logger.error(f"Error during ingest command: {e}", exc_info=True)
             sys.exit(1)


# Use asyncio to run the main async function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())