import chainlit as cl
import httpx
import logging
import os
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('ChainlitUI')

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
logger.info(f"Configured RAG backend URL: {BACKEND_URL}")

# --- Helper for sending messages in Chainlit ---
async def send_status_message(content: str, author: str = "System", type: str = "info"):
    """Helper to send status messages in Chainlit."""
    icon_map = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
    full_content = f"{icon_map.get(type, '➡️')} {content}"
    await cl.Message(author=author, content=full_content).send()


# --- Check Backend Status ---
async def check_backend_status():
    logger.info(f"Checking backend status at {BACKEND_URL}/status")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/status", timeout=5.0)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            status_data = response.json()
            logger.info(f"Backend status: {status_data}")
            return status_data
    except httpx.RequestError as e:
        logger.critical(f"Backend connection error: {e}", exc_info=True)
        return {"status": "Error", "message": f"Backend connection failed: {e}"}
    except httpx.HTTPStatusError as e:
        logger.critical(f"Backend returned error status: {e.response.status_code} - {e.response.text}", exc_info=True)
        return {"status": "Error", "message": f"Backend returned error status {e.response.status_code}: {e.response.text[:200]}..."}
    except Exception as e:
        logger.critical(f"An unexpected error occurred checking backend status: {e}", exc_info=True)
        return {"status": "Error", "message": f"Unexpected error checking backend status: {e}"}


# --- Chainlit Event Handlers ---
@cl.on_chat_start
async def start():
    logger.info("Chainlit chat session started.")
    await cl.Message(content="Connecting to RAG backend and initializing...").send()

    status = await check_backend_status()
    cl.user_session.set("backend_status", status)

    if status.get("status") == "Running" and status.get("rag_orchestrator_status") == "Initialized":
        await send_status_message("✅ Backend connected and RAG system initialized.", type="success")

        loaded_count = status.get("loaded_sources_count", 0)
        if loaded_count > 0:
            await send_status_message(f"📚 {loaded_count} document source(s) are already loaded from a previous run.", type="info")
            if status.get("loaded_sources", []):
                source_list_str = "\n".join([f"- {src}" for src in status['loaded_sources']])
                await cl.Message(content=f"Loaded sources:\n{source_list_str}").send()

            await cl.Message(content="You can now ask questions about the documents!").send()

        else:
            logger.info(f"Triggering ingestion via backend at {BACKEND_URL}/ingest")
            ingest_status_message = cl.Message(content="📂 Starting document ingestion...")
            await ingest_status_message.send()

            try:
                async with httpx.AsyncClient() as client:
                    ingest_response = await client.post(f"{BACKEND_URL}/ingest", json={"pdf_folder_path": "pdf"}, timeout=600.0)
                    ingest_response.raise_for_status()
                    ingest_result = ingest_response.json()

                logger.info(f"Backend ingestion result: {ingest_result}")

                if ingest_result.get("status") == "Success":
                    ingest_status_message.content = f"📚 Document ingestion complete! {ingest_result.get('message', '')}"
                    await ingest_status_message.update()

                    updated_status = await check_backend_status()
                    cl.user_session.set("backend_status", updated_status)

                    loaded_count = updated_status.get("loaded_sources_count", 0)
                    if loaded_count > 0:
                        await cl.Message(content=f"✅ {loaded_count} document source(s) are loaded and ready for querying.").send()
                        if updated_status.get("loaded_sources", []):
                            source_list_str = "\n".join([f"- {src}" for src in updated_status['loaded_sources']])
                            await cl.Message(content=f"Loaded sources:\n{source_list_str}").send()
                        await cl.Message(content="You can now ask questions about the documents!").send()
                    else:
                        await cl.Warning(content="⚠️ Ingestion finished, but no document sources were successfully loaded.").send()
                        await cl.Message(content="You can still ask questions, but answers will not be based on documents.").send()

                else:
                    ingest_status_message.content = f"❌ Document ingestion failed on backend: {ingest_result.get('message', 'Unknown error')}"
                    await ingest_status_message.update()
                    await send_status_message("Ingestion failed. Please check backend logs.", type="error")

            except httpx.RequestError as e:
                logger.critical(f"Ingestion request failed: {e}", exc_info=True)
                ingest_status_message.content = f"❌ Failed to send ingestion request to backend: {e}"
                await ingest_status_message.update()
                await send_status_message("Ingestion request failed. Please check backend logs.", type="error")
            except httpx.HTTPStatusError as e:
                logger.critical(f"Backend ingestion returned error status: {e.response.status_code} - {e.response.text}", exc_info=True)
                ingest_status_message.content = f"❌ Backend ingestion failed (Status {e.response.status_code}): {e.response.text[:200]}..."
                await ingest_status_message.update()
                await send_status_message("Backend ingestion failed. Please check backend logs.", type="error")
            except Exception as e:
                logger.critical(f"An unexpected error occurred during ingestion: {e}", exc_info=True)
                ingest_status_message.content = f"❌ An unexpected error occurred during ingestion: {e}"
                await ingest_status_message.update()
                await send_status_message("An unexpected error occurred during ingestion. Please check backend logs.", type="error")

    else:
        await send_status_message("❌ Backend not initialized or unreachable. Please check server logs and ensure the backend is running correctly.", type="error")


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message from user: {message.content}")

    backend_status = cl.user_session.get("backend_status")

    if not backend_status or backend_status.get("status") != "Running":
         logger.warning("Backend not running according to status in session. Skipping query.")
         await cl.Warning(content="⚠️ RAG system backend is not running or not initialized. Please refresh the page.").send()
         return

    if backend_status.get("retriever_status") != "Ready" or backend_status.get("loaded_sources_count", 0) == 0:
         logger.warning("Backend retriever not ready or no sources loaded according to status. Skipping RAG query.")
         await cl.Warning(content="⚠️ Documents were not loaded successfully. RAG queries cannot be processed based on documents.").send()
         return

    thinking_message = cl.Message(content="💭 Thinking...")
    await thinking_message.send()

    logger.info(f"Sending query to backend: {message.content}")

    try:
        async with httpx.AsyncClient() as client:
            query_response = await client.post(
                f"{BACKEND_URL}/query",
                json={"query": message.content},
                timeout=300.0
            )
            query_response.raise_for_status()
            answer_data = query_response.json()

        logger.info(f"Received response from backend (snippet): {answer_data.get('answer', '')[:100]}...")

        answer = answer_data.get("answer", "Backend did not return a valid answer.")
        # sources_info = answer_data.get("sources", []) # Uncomment if backend returns sources

        thinking_message.content = answer

        # Add source elements if backend provides them
        # elements = []
        # if sources_info:
        #     source_elements = [
        #         cl.Text(name=f"Source {i+1}", content=str(source_info))
        #         for i, source_info in enumerate(sources_info)
        #     ]
        #     elements.extend(source_elements)
        #     thinking_message.content += "\n\n**Sources:**"
        #
        # thinking_message.elements = elements if elements else None

        await thinking_message.update()

        logger.info("Response sent to UI.")


    except httpx.RequestError as e:
        logger.critical(f"Query request failed: {e}", exc_info=True)
        await thinking_message.remove()
        await send_status_message(f"Failed to send query request to backend: {e}. Please check backend logs.", type="error")

    except httpx.HTTPStatusError as e:
        logger.critical(f"Backend query returned error status: {e.response.status_code} - {e.response.text}", exc_info=True)
        await thinking_message.remove()
        await send_status_message(f"Backend query failed (Status {e.response.status_code}): {e.response.text[:200]}... Please check backend logs.", type="error")

    except Exception as e:
        logger.error(f"An unexpected error occurred during query processing: {e}", exc_info=True)
        await thinking_message.remove()
        await send_status_message(f"An unexpected error occurred during query processing: {e}. Please check backend logs.", type="error")
        
# Note: Remember to run your FastAPI backend separately first:
# uvicorn fastapi_backend:app --reload --port 8000
# Then run the Chainlit UI:
# chainlit run ui.py --port 8002