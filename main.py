import logging
import os
import sys
from typing import List

# Import the orchestrator from the doc_rag module
from doc_rag import ManagerRAG

# Import utilities from utils.py (this will configure logging and initialize clients)
import utils

# Main logger for this file's execution flow
logger = logging.getLogger('Main')


# --- Main Execution ---
def main():
    """Main entry point for the RAG application."""
    logger.info("Starting RAG application")
    try:
        logger.info("Initializing ManagerRAG Orchestrator.")
        try:
             orchestrator = ManagerRAG()
             logger.info("ManagerRAG Orchestrator initialized.")
        except Exception as e:
             logger.critical(f"Failed to initialize ManagerRAG Orchestrator: {e}", exc_info=True)
             print(f"\nA critical error occurred during initialization: {e}")
             sys.exit(1)

        pdf_folder_path = "pdf"
        if not os.path.isdir(pdf_folder_path):
            logger.critical(f"PDF folder '{pdf_folder_path}' not found.")
            print(f"\nError: PDF folder '{pdf_folder_path}' not found. Please create it.")
            sys.exit(1)


        logger.info(f"Starting document ingestion for PDFs in folder '{pdf_folder_path}'.")
        try:
            orchestrator.ingest_pdfs_from_folder(pdf_folder_path)
            logger.info("Document ingestion completed.")
        except FileNotFoundError as e:
             logger.critical(f"Ingestion failed: {e}")
             print(f"\nError: {e}")
             sys.exit(1)
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            print(f"\nError: Document ingestion failed: {e}")


        if orchestrator.retriever is not None:
            logger.info(f"Starting query phase with {len(orchestrator.loaded_sources)} loaded sources.")
            questions: List[str] = [
                'What is Rdc maximum of VCHA042A-100MS62M?',
                'what is max ramp up rate of power choke coil?',
                'What is printed coverage area of RHA00400L?',
            ]
            for i, question in enumerate(questions, 1):
                 logger.info(f"Processing question {i}/{len(questions)}: {question}")
                 answer = orchestrator.query(question, history=[]) # Pass an empty list for history
                 logger.info(f"Answer for question {i} (snippet): {answer[:100]}...")
                 print(f"\n--- Question {i}: {question} ---\n")
                 print(f"Answer: {answer}\n")
                 print("-" * 20)
        else:
            logger.warning("Skipping query phase because no sources were successfully loaded or indexed.")
            print("\nSkipping query phase because document ingestion failed or didn't produce usable data.")


    except Exception as e:
        logger.critical(f"Application failed during execution: {str(e)}", exc_info=True)
        print(f"\nA critical error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()