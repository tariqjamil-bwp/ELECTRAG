import logging
import os
import sys
from typing import Dict, List, Optional, Any

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# httpx for making requests (potentially needed for backend-to-backend or testing)
# import httpx

# Import your core RAG components
# Note: Importing utils here will configure logging and initialize models/clients
import utils # This import runs code in utils.py
from doc_rag import ManagerRAG # Import your orchestrator class

# --- Logging ---
# Logging is already configured in utils.py upon import
logger = logging.getLogger('FastAPIBackend')
logger.info("FastAPI backend starting up.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="RAG Backend API",
    description="API for Document Ingestion and Question Answering with RAG.",
    version="0.1.0",
)

# --- RAG Orchestrator Instance ---
# Instantiate the RAG orchestrator when the FastAPI app starts up.
# This instance will be used across all requests.
rag_orchestrator: Optional[ManagerRAG] = None

@app.on_event("startup")
async def startup_event():
    """Actions to perform when the FastAPI application starts."""
    logger.info("FastAPI startup event triggered.")
    global rag_orchestrator
    try:
        # Initialize the RAG orchestrator
        # This will internally initialize DocLoader, DocEmbedder, DocQA, DocRanker
        # and the DocEmbedder model dimension.
        rag_orchestrator = ManagerRAG()
        logger.info("RAG Orchestrator initialized successfully.")

        # You might want to trigger ingestion automatically on startup,
        # or wait for a specific /ingest endpoint call.
        # For simplicity, let's keep ingestion triggered by the /ingest endpoint.

    except Exception as e:
        logger.critical(f"Failed to initialize RAG orchestrator on startup: {e}", exc_info=True)
        # The app will still run, but /query and /ingest will return errors.
        # Consider adding a check in endpoints to ensure rag_orchestrator is not None.


@app.on_event("shutdown")
def shutdown_event():
    """Actions to perform when the FastAPI application shuts down."""
    logger.info("FastAPI shutdown event triggered.")
    # Add any cleanup logic here if needed (e.g., saving index state if not using manifest)
    # The current manifest/cache handling is file-based and doesn't need explicit shutdown save
    pass


# --- API Endpoints ---

@app.get("/status")
async def get_status():
    """Returns the current status of the backend and RAG orchestrator."""
    logger.info("Received GET /status request.")
    status = {
        "status": "Running",
        "message": "RAG backend API is running.",
        "rag_orchestrator_status": "Not Initialized",
        "loaded_sources_count": 0,
        "retriever_status": "Not Ready",
        "loaded_sources": [] # Include list of sources
    }

    if rag_orchestrator is not None:
        status["rag_orchestrator_status"] = "Initialized"
        status["loaded_sources_count"] = len(rag_orchestrator.loaded_sources)
        if rag_orchestrator.retriever is not None:
            status["retriever_status"] = "Ready"

        # Provide a list of loaded source identifiers
        if rag_orchestrator.loaded_sources:
             # Use basename for a cleaner UI display
             status["loaded_sources"] = [os.path.basename(path) for path in rag_orchestrator.loaded_sources.keys()]

    logger.info(f"Responding to /status with: {status}")
    return status

# Request model for ingestion
class IngestRequest(BaseModel):
    pdf_folder_path: str = "pdf" # Default to the 'pdf' folder


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Triggers the document ingestion process from a specified PDF folder."""
    logger.info(f"Received POST /ingest request for folder: {request.pdf_folder_path}")

    if rag_orchestrator is None:
        logger.error("Ingestion requested, but RAG orchestrator is not initialized.")
        raise HTTPException(status_code=500, detail="RAG orchestrator is not initialized.")

    try:
        # Ingestion can be time-consuming. In a real application, this might be better
        # as an asynchronous task or using background workers (like Celery/RQ).
        # For this example, it runs synchronously within the request.
        # The tqdm in doc_processing/doc_rag will print to the backend console.
        rag_orchestrator.ingest_pdfs_from_folder(request.pdf_folder_path)

        logger.info("Document ingestion process completed via API.")

        # Update status after ingestion
        ingestion_status = "Success"
        message = "Document ingestion completed."

        # Re-check status to include updated loaded sources count etc.
        updated_status = {
            "status": "Success",
            "message": message,
            "rag_orchestrator_status": "Initialized",
            "loaded_sources_count": len(rag_orchestrator.loaded_sources),
            "retriever_status": "Ready" if rag_orchestrator.retriever is not None else "Not Ready",
            "loaded_sources": [os.path.basename(p) for p in rag_orchestrator.loaded_sources.keys()] # List loaded sources
        }

        return updated_status

    except FileNotFoundError as e:
        logger.error(f"Ingestion failed (folder not found): {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"PDF folder not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}", exc_info=True)
        # Return a 500 Internal Server Error for other exceptions
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {e}")


# Request model for query
class QueryRequest(BaseModel):
    query: str

# Response model for query (optional but good practice)
class QueryResponse(BaseModel):
     answer: str
     # sources: List[Dict[str, Any]] # Uncomment if your pipeline returns source details


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Processes a user query using the loaded RAG system."""
    logger.info(f"Received POST /query request: {request.query[:100]}...")

    if rag_orchestrator is None:
        logger.error("Query requested, but RAG orchestrator is not initialized.")
        raise HTTPException(status_code=500, detail="RAG orchestrator is not initialized.")

    # Check if retrieval is possible (documents loaded and indexed)
    if rag_orchestrator.retriever is None:
         logger.warning("Query requested, but RAG retriever is not ready (ingestion likely failed or no docs).")
         # Return a specific message indicating no documents
         return {"answer": "Documents are not loaded or indexed. Cannot answer based on documents.", "sources": []}


    try:
        # The query method handles the RAG pipeline (retrieval, QA, ranking)
        # It returns the final answer string.
        # Modify ManagerRAG.query if you want it to return sources too.
        answer = rag_orchestrator.query(request.query)

        logger.info(f"Query processing completed. Answer snippet: {answer[:100]}...")

        # Structure the response as expected by the frontend
        response_data = {
            "answer": answer,
            # Add sources here if the query method was modified to return them
            # "sources": sources
        }

        return response_data

    except Exception as e:
        logger.error(f"An error occurred during query processing: {e}", exc_info=True)
        # Return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {e}")