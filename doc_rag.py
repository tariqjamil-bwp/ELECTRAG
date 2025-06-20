import logging
import os
import sys
from typing import List, Dict, Optional
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import re
from tqdm import tqdm # Keep tqdm for console progress
from datetime import datetime

# Import components and helpers from doc_processing
from doc_processing import (
    DocLoader,
    DocEmbedder,
    DocRetriever,
    # Moved persistence constants/helpers to utils
)

# Import utilities from utils.py
from utils import (
    model_manager,
    EMBEDDING_MODEL,
    ENC,
    num_tokens,
    _cleanup_cache_files, # Keep for use in ManagerRAG
    LLM_MODELS, # Needed for ranking fallback
    VECTORDB_DIR, # Persistence constants needed here
    MANIFEST_PATH,
    calculate_file_hash,
    load_manifest,
    save_manifest,
)
import json

# --- QA Agent ---
class DocQA:
    def __init__(self):
        self.logger = logging.getLogger('DocQA')
        self.logger.info("Initialized DocQA.")

    # Modified answer to accept history
    def answer(self, question: str, context: List[str], history: List[Dict]) -> str:
        self.logger.debug(f"Generating answer for question: {question[:100]}... with {len(context)} context chunks.")
        if not context:
            self.logger.warning("No context provided for answering question.")
            return "I cannot answer the question without relevant context."

        try:
            context_str = '---\n'.join(context)
            MAX_CONTEXT_TOKENS = 7000 # Adjust based on model capacity and other prompt parts
            context_tokens = ENC.encode(context_str)
            if len(context_tokens) > MAX_CONTEXT_TOKENS:
                 self.logger.warning(f"Context token count ({len(context_tokens)}) exceeds MAX_CONTEXT_TOKENS ({MAX_CONTEXT_TOKENS}). Truncating context.")
                 context_str = ENC.decode(context_tokens[:MAX_CONTEXT_TOKENS]) + "\n[... Context truncated ...]"

            # Format history for prompt
            history_str = ""
            # We typically want the LLM to see the history in chronological order, ending with the last assistant turn
            # before the current user question. The history passed here includes the current user message.
            # Let's format all but the *last* message (the current user question) as history.
            # The last message (current question) is already in the 'question' variable.
            # Format history for prompt (excluding the last message which is handled separately)
            history_for_prompt = history[:-1] # Exclude the last message (current user question)
            if history_for_prompt:
                history_str = "Chat History:\n"
                for turn in history_for_prompt:
                    # Capitalize role for prompt clarity
                    history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"
                history_str += "---\n\n" # Separator


            prompt = (
                "You are an expert assistant. Use the following context and chat history to answer the question.\n\n"
                f"{history_str}" # Include history
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\nAnswer:"
            )

            client = model_manager.get_client()
            model_name = model_manager.get_current_model_name()
            self.logger.debug(f"Using model '{model_name}' for answer generation.")

            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}], # Sending as a single user message for simplicity
                temperature=0.2,
                max_tokens=500,
            )
            answer = resp.choices[0].message.content.strip()
            self.logger.debug(f"Generated answer (snippet): {answer[:100]}...")
            return answer
        except Exception as e:
             self.logger.error(f"Error calling API for answer generation: {type(e).__name__}: {e}", exc_info=True)
             raise # Re-raise to answer_parallel

    # Modified answer_parallel to accept history and pass it to answer
    def answer_parallel(self, question: str, candidate_contexts: List[List[str]], history: List[Dict], max_retries_per_set: int = 2) -> List[str]:
        self.logger.info(f"Generating parallel answers for {len(candidate_contexts)} candidate contexts sets.")
        if not candidate_contexts:
             self.logger.warning("No candidate contexts sets provided for parallel answering.")
             return []

        results = [None] * len(candidate_contexts)
        futures_queue = []

        try:
            max_workers = min(len(candidate_contexts), os.cpu_count() or 4)
            self.logger.debug(f"Using ThreadPoolExecutor with max_workers={max_workers}.")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, context_set in enumerate(candidate_contexts):
                    # Pass history to the answer function
                    future = executor.submit(self.answer, question, context_set, history)
                    futures_queue.append((future, idx, 0)) # Store (future, original_index, retry_count)

                # Using tqdm for console progress
                pbar = tqdm(total=len(candidate_contexts), desc="Generating answers")
                completed_count = 0

                while completed_count < len(candidate_contexts):
                    done_futures = [f for f in futures_queue if f[0].done()]
                    futures_queue = [f for f in futures_queue if not f[0].done()]

                    if not done_futures:
                         if futures_queue:
                              import time
                              time.sleep(0.1)
                         continue


                    for future, original_idx, retry_count in done_futures:
                        try:
                            ans = future.result()
                            results[original_idx] = ans
                            completed_count += 1
                            pbar.update(1) # Update tqdm progress bar
                            self.logger.debug(f"Completed answer for candidate context set index {original_idx} after {retry_count} retries.")

                        except Exception as e:
                            self.logger.error(f"Attempt {retry_count+1} for answer index {original_idx} failed: {type(e).__name__}: {e}", exc_info=True)
                            if retry_count < max_retries_per_set:
                                self.logger.warning(f"Attempting to cycle model and retry answer index {original_idx}.")
                                if model_manager.handle_api_error(e):
                                    self.logger.warning(f"Retrying answer index {original_idx} with model {model_manager.get_current_model_name()} (Retry {retry_count+1}/{max_retries_per_set}).")
                                    # Pass history again on retry
                                    new_future = executor.submit(self.answer, question, candidate_contexts[original_idx], history)
                                    futures_queue.append((new_future, original_idx, retry_count + 1)) # Add back to queue
                                else:
                                    self.logger.critical(f"Model cycling failed for answer index {original_idx}. Giving up after {retry_count+1} attempts.")
                                    results[original_idx] = f"Error generating answer after multiple model retries: {e}"
                                    completed_count += 1
                                    pbar.update(1)
                            else:
                                self.logger.error(f"Answer index {original_idx} failed after max retries ({max_retries_per_set}). Giving up.")
                                results[original_idx] = f"Error generating answer after max retries: {e}"
                                completed_count += 1
                                pbar.update(1)

                if completed_count < len(candidate_contexts) and not done_futures:
                     import time
                     time.sleep(0.1)


            pbar.close() # Close tqdm progress bar

            final_answers = [ans if ans is not None else "Error: Answer generation did not complete unexpectedly" for ans in results]

            self.logger.info(f"Completed parallel answer generation for {len(final_answers)} contexts sets.")
            return final_answers

        except Exception as e:
            self.logger.error(f"Failed during parallel answer generation setup or processing: {str(e)}", exc_info=True)
            raise


# --- Ranking Agent ---
class DocRanker:
    def __init__(self):
        self.logger = logging.getLogger('DocRanker')
        self.logger.info("Initialized DocRanker.")

    # Modified rank to accept history
    def rank(self, question: str, candidate_answers: List[str], candidate_contexts: List[List[str]], history: List[Dict], max_retries: int = 2) -> tuple[str, int]:
        self.logger.info(f"Ranking {len(candidate_answers)} candidate answers for question: {question[:100]}...")
        if not candidate_answers:
             self.logger.warning("No candidate answers provided for ranking.")
             return "No answers available.", -1
        if len(candidate_answers) != len(candidate_contexts):
             self.logger.error(f"Mismatch between number of answers ({len(candidate_answers)}) and contexts ({len(candidate_contexts)}) provided for ranking.")
             return "Ranking failed due to data mismatch.", -1

        valid_candidates = [(ans, ctx) for ans, ctx in zip(candidate_answers, candidate_contexts)
                            if ans and not ans.startswith("Error generating answer:") and not ans.startswith("Internal error generating answer:")]
        if not valid_candidates:
            self.logger.warning("All candidate answers were error messages or empty. Cannot rank valid candidates.")
            return candidate_answers[0] if candidate_answers else "No valid answers to rank.", -1


        self.logger.debug(f"Ranking among {len(valid_candidates)} valid candidates.")

        attempt = 0
        while attempt <= max_retries:
            try:
                summary_list = []
                MAX_SNIPPET_LEN = 200
                # Only include valid candidates in the prompt summary
                for original_idx, (ans, ctx) in enumerate(zip(candidate_answers, candidate_contexts)):
                     if ans and not ans.startswith("Error generating answer:") and not ans.startswith("Internal error generating answer:"):
                         ctx_snippet = (ctx[0][:MAX_SNIPPET_LEN] + '...') if ctx and ctx[0] else 'No context'
                         ans_snippet = ans[:MAX_SNIPPET_LEN] + '...' if ans else 'No answer'
                         summary_list.append(f"Candidate #{original_idx + 1}:\nContext Snippet: {ctx_snippet}\nAnswer Snippet: {ans_snippet}")

                summary = "\n\n---\n\n".join(summary_list)

                # Format history for prompt
                history_str = ""
                # Again, format all but the *last* message (the current user question) as history.
                history_for_prompt = history[:-1]
                if history_for_prompt:
                     history_str = "Chat History:\n"
                     for turn in history_for_prompt:
                         history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"
                     history_str += "---\n\n" # Separator


                ranking_prompt = (
                    "You are a ranking expert. Below are several candidate answers and snippets of the context used "
                    "to generate them for the given question.\n\n"
                    f"{history_str}" # Include history
                    "Analyze each candidate answer snippet and its context snippet. "
                    "Determine which *original candidate answer* (by its number #) is the best "
                    "based on relevance, accuracy, completeness, and clarity. Provide a brief reason for choosing the best answer. "
                    "Finally, state the number of the best original candidate answer and the full text of that answer.\n\n"
                    f"Question: {question}\n\n"
                    f"Candidate Answer Snippets and Context Snippets:\n{summary}\n\n"
                    "Format your response strictly as follows:\n"
                    "Candidate #[number]\n"
                    "Reason: [brief explanation]\n"
                    "Best Answer:\n[The full text of the *original* best answer]"
                )

                client = model_manager.get_client()
                model_name = model_manager.get_current_model_name()
                self.logger.debug(f"Using model '{model_name}' for ranking generation (Attempt {attempt+1}/{max_retries+1}).")

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": ranking_prompt}],
                    temperature=0.2,
                    max_tokens=500,
                )
                response_text = resp.choices[0].message.content.strip()
                self.logger.debug(f"LLM ranking response received (Attempt {attempt+1}): {response_text[:100]}...")

                m = re.search(r"Candidate #(\d+)\s*\nReason:(.*?)\n+Best Answer:\n(.+)", response_text, re.DOTALL)

                if m:
                    try:
                        cand_idx_str = m.group(1).strip()
                        llm_chosen_idx = int(cand_idx_str) - 1
                        reason = m.group(2).strip()

                        if 0 <= llm_chosen_idx < len(candidate_answers):
                             if candidate_answers[llm_chosen_idx] in ["Error generating answer:", "Internal error generating answer:"]:
                                  self.logger.warning(f"LLM selected candidate #{llm_chosen_idx+1} which was an error message. Treating as model error (Attempt {attempt+1}).")
                                  if model_manager.handle_api_error(ValueError(f"LLM selected error candidate {llm_chosen_idx}")):
                                       attempt += 1
                                       continue
                                  else:
                                       self.logger.error("Model cycling failed after LLM selected error candidate. Giving up.")
                                       break
                             else:
                                 final_answer = candidate_answers[llm_chosen_idx]
                                 chosen_idx = llm_chosen_idx
                                 self.logger.info(f"Ranking successful. LLM selected valid candidate #{llm_chosen_idx+1} (Attempt {attempt+1}). Reason: {reason}")
                                 return final_answer, chosen_idx


                        else:
                            self.logger.warning(f"LLM selected candidate #{llm_chosen_idx+1}, which is out of bounds (0-{len(candidate_answers)-1}) (Attempt {attempt+1}).")
                            if model_manager.handle_api_error(ValueError(f"LLM returned out-of-bounds index {llm_chosen_idx}")):
                                attempt += 1
                                continue
                            else:
                                self.logger.error("Model cycling failed after out-of-bounds index from LLM. Giving up.")
                                break

                    except ValueError:
                         self.logger.error(f"Could not parse valid candidate number '{cand_idx_str}' from LLM response (Attempt {attempt+1}).", exc_info=True)
                         if model_manager.handle_api_error(ValueError(f"Failed to parse LLM response: {response_text[:100]}...")):
                             attempt += 1
                             continue
                         else:
                            self.logger.error("Model cycling failed after parsing error. Giving up.")
                            break

                else:
                    self.logger.warning(f"Could not parse ranking output using regex (Attempt {attempt+1}). Response did not match expected format.")
                    if model_manager.handle_api_error(ValueError(f"LLM response format mismatch: {response_text[:100]}...")):
                         attempt += 1
                         continue
                    else:
                         self.logger.error("Model cycling failed after format mismatch. Giving up.")
                         break

            except (APIStatusError, APIConnectionError, RateLimitError) as api_exc:
                self.logger.error(f"API error during ranking (Attempt {attempt+1}): {type(api_exc).__name__}: {api_exc}", exc_info=True)
                if model_manager.handle_api_error(api_exc):
                    attempt += 1
                    continue

        # --- Fallback if retries exhausted or fatal error ---
        # This block is now correctly outside the while loop
        self.logger.error(f"Ranking failed after {max_retries+1} attempts.")
        if valid_candidates:
            self.logger.info("Returning first valid candidate answer as fallback after ranking failure.")
            fallback_ans = valid_candidates[0][0]
            try:
                 fallback_idx = candidate_answers.index(fallback_ans)
                 return fallback_ans, fallback_idx
            except ValueError:
                 self.logger.error("Consistency error: First valid candidate not found in original list.")
                 return fallback_ans, -1
        else:
            self.logger.warning("No valid candidate answers available for fallback after ranking failure.")
            return "Ranking failed and no valid candidate answers were available.", -1


class ManagerRAG:
    def __init__(self, n_candidates=3, k=5):
        self.logger = logging.getLogger('ManagerRAG')
        self.logger.info(f"Initializing ManagerRAG with n_candidates={n_candidates}, k={k}")

        try:
            self.loader = DocLoader()
            self.embedding_agent_model = DocEmbedder(model_name=EMBEDDING_MODEL)
            self.qa = DocQA()
            self.ranker = DocRanker()
            self.logger.info("ManagerRAG components initialized.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize ManagerRAG components: {e}", exc_info=True)
            raise

        self.loaded_sources: Dict[str, Dict] = {}
        self.retriever: Optional[DocRetriever] = None
        self.n_candidates = n_candidates
        self.k = k
        self.logger.info("ManagerRAG initialized.")


    def ingest_pdfs_from_folder(self, pdf_folder_path):
        self.logger.info(f"Starting ingestion process for PDFs in folder: {pdf_folder_path}")
        if not os.path.isdir(pdf_folder_path):
            self.logger.critical(f"PDF folder not found or is not a directory: {pdf_folder_path}")
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")

        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            self.logger.warning(f"No PDF files found in folder: {pdf_folder_path}")
            self.loaded_sources = {}
            self.retriever = None
            return

        self.logger.info(f"Found {len(pdf_files)} PDF files in folder.")
        self.loaded_sources = {}
        manifest = load_manifest(MANIFEST_PATH)
        updated_manifest = manifest.copy()

        # Using tqdm for console progress
        for pdf_filename in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(pdf_folder_path, pdf_filename)
            pdf_path_abs = os.path.abspath(pdf_path)
            self.logger.info(f"Processing file: {pdf_filename}")

            try: # Try block for processing a single file
                current_hash = calculate_file_hash(pdf_path_abs)
                self.logger.debug(f"Calculated hash for {pdf_filename}: {current_hash}")

                pdf_base = os.path.basename(pdf_path_abs)
                sanitized_name = re.sub(r'[^\w\s.-]', '_', os.path.splitext(pdf_base)[0])
                unique_id = current_hash[:8]
                file_base_name = f"{sanitized_name}_{unique_id}"
                index_filename = f"{file_base_name}.faiss"
                texts_filename = f"{file_base_name}.json"
                index_path_cache = os.path.join(VECTORDB_DIR, index_filename)
                texts_path_cache = os.path.join(VECTORDB_DIR, texts_filename)

                cache_hit = False
                cached_info = manifest.get(pdf_path_abs)

                if cached_info and cached_info.get("hash") == current_hash:
                    manifest_index_path = cached_info.get("index_path")
                    manifest_texts_path = cached_info.get("texts_path")

                    if manifest_index_path and manifest_texts_path and os.path.exists(manifest_index_path) and os.path.exists(manifest_texts_path):
                        self.logger.info(f"Cache hit for {pdf_filename}. Loading...")
                        try: # Try block for loading from cache
                            file_embedder = DocEmbedder(model_name=self.embedding_agent_model.model_name, batch_size=self.embedding_agent_model.batch_size)
                            file_embedder.load_index(manifest_index_path)

                            with open(manifest_texts_path, 'r', encoding='utf-8') as f:
                                texts_list = json.load(f)
                            self.logger.info(f"Loaded {len(texts_list)} text chunks from {manifest_texts_path}.")

                            self.loaded_sources[pdf_path_abs] = {
                                'embedder': file_embedder,
                                'texts': texts_list
                            }
                            self.logger.debug(f"Added '{pdf_filename}' to loaded sources from cache.")
                            cache_hit = True
                        except Exception as e:
                            self.logger.error(f"Failed to load cache for {pdf_filename}: {e}. Will re-process.", exc_info=True)
                            if pdf_path_abs in updated_manifest:
                                _cleanup_cache_files(updated_manifest[pdf_path_abs].get("index_path"), updated_manifest[pdf_path_abs].get("texts_path"))
                                del updated_manifest[pdf_path_abs]
                            cache_hit = False
                    else:
                         self.logger.info(f"Cache miss for {pdf_filename}. Cache files not found.")
                         if pdf_path_abs in updated_manifest:
                             _cleanup_cache_files(updated_manifest[pdf_path_abs].get("index_path"), updated_manifest[pdf_path_abs].get("texts_path"))
                             del updated_manifest[pdf_path_abs]
                         cache_hit = False
                else:
                    self.logger.info(f"Cache miss for {pdf_filename}. No entry found or hash mismatch.")
                    if pdf_path_abs in updated_manifest:
                         self.logger.info(f"Hash mismatch for {pdf_filename}. Re-processing.")
                         old_index_path = updated_manifest[pdf_path_abs].get("index_path")
                         old_texts_path = updated_manifest[pdf_path_abs].get("texts_path")
                         _cleanup_cache_files(old_index_path, old_texts_path)
                         del updated_manifest[pdf_path_abs]

                    cache_hit = False


                if not cache_hit:
                    self.logger.info(f"Processing {pdf_filename} (cache miss).")
                    try: # Try block for processing and saving
                        texts_list = self.loader.load_and_split(pdf_path_abs)
                        if not texts_list:
                            self.logger.warning(f"No text chunks extracted from {pdf_filename}. Skipping.")
                            continue

                        file_embedder = DocEmbedder(model_name=self.embedding_agent_model.model_name, batch_size=self.embedding_agent_model.batch_size)
                        file_embedder.create_new_index()
                        file_embedder.add_to_index(texts_list) # This calls embed which uses tqdm

                        self.logger.info(f"Embeddings added to new index for {pdf_filename}. Index size: {file_embedder.index.ntotal}")

                        try: # Inner try for saving cache
                            file_embedder.save_index(index_path_cache)
                            with open(texts_path_cache, 'w', encoding='utf-8') as f:
                                json.dump(texts_list, f)
                            self.logger.info(f"Saved FAISS index to {index_path_cache} and texts to {texts_path_cache} for {pdf_filename}")

                            updated_manifest[pdf_path_abs] = {
                                "hash": current_hash,
                                "index_path": index_path_cache,
                                "texts_path": texts_path_cache,
                                "timestamp": datetime.now().isoformat()
                            }
                            self.logger.debug("Manifest updated with new cache entry for {pdf_filename}.")

                        except Exception as save_e:
                            self.logger.error(f"Failed to save cache for {pdf_filename}: {save_e}. This file will be re-processed next time.", exc_info=True)
                            # Saving failed, but processing was successful, continue processing other files
                            # The entry is NOT added to updated_manifest

                        # Add to loaded_sources ONLY if processing was successful and there are chunks
                        if file_embedder.index is not None and file_embedder.index.ntotal > 0 and texts_list:
                             self.loaded_sources[pdf_path_abs] = {
                                'embedder': file_embedder,
                                'texts': texts_list
                             }
                             self.logger.debug(f"Added processed '{pdf_filename}' to loaded sources.")
                        else:
                            if texts_list:
                                self.logger.warning(f"Index was empty after processing {pdf_filename}. Not adding to loaded sources.")

                    except Exception as process_e:
                        self.logger.error(f"Failed during processing (load/split/embed) for {pdf_filename}: {str(process_e)}", exc_info=True)
                        continue # Continue outer loop

            except Exception as file_loop_e:
                 # Catch any unexpected error during the initial checks or hash calculation for this file
                 self.logger.error(f"Unexpected error during file loop processing for {pdf_filename}: {file_loop_e}", exc_info=True)
                 continue # Continue outer loop

        # tqdm progress bar finishes automatically after the loop

        self.logger.info("Saving updated manifest.")
        save_manifest(MANIFEST_PATH, updated_manifest)
        self.logger.info("Manifest save complete.")

        if self.loaded_sources:
             self.retriever = DocRetriever(self.embedding_agent_model, self.loaded_sources)
             self.logger.info(f"DocRetriever initialized with {len(self.loaded_sources)} sources.")
        else:
             self.logger.warning("No sources successfully loaded/processed. DocRetriever will not be initialized.")
             self.retriever = None


    # Modified query to accept history
    def query(self, question: str, history: List[Dict]):
        self.logger.info(f"Starting query process for question: {question[:100]}...")
        if self.retriever is None:
             self.logger.error("DocRetriever is not initialized. Document ingestion likely failed or hasn't occurred yet, or no valid PDFs were processed.")
             return "Error: DocRetriever is not initialized. Documents may not be loaded."
        if not self.loaded_sources:
             self.logger.warning("No loaded sources available to query.")
             return "No relevant information found (no documents loaded)."

        total_indexed_vectors = sum(source_info['embedder'].index.ntotal for source_info in self.loaded_sources.values()
                                    if source_info.get('embedder') and source_info['embedder'].index)
        if total_indexed_vectors == 0:
             self.logger.warning("All loaded source indexes are empty. Cannot retrieve context for query.")
             return "No relevant information found (all document indexes are empty)."


        try:
            candidates = self.retriever.retrieve_candidates(
                question, n_candidates=self.n_candidates, k=self.k)
            self.logger.info(f"Retrieval returned {len(candidates)} non-empty candidate sets.")

            if not candidates:
                 self.logger.warning("No candidates retrieved for the query from any source.")
                 return "No relevant information found for the query."

            # Pass history to answer_parallel
            candidate_answers = self.qa.answer_parallel(question, candidates, history)
            self.logger.info(f"DocQA returned {len(candidate_answers)} candidate answers.")

            # Pass history to rank
            final_answer, chosen_idx = self.ranker.rank(question, candidate_answers, candidates, history)
            self.logger.info(f"Ranking completed. Chosen answer index: {chosen_idx}.")

            self.logger.info(f"Final answer generated (snippet): {final_answer[:100]}...")

            return final_answer
        except Exception as e:
            self.logger.error(f"Failed to process query '{question[:100]}...': {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"