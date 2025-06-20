import logging
import faiss
import numpy as np
import ollama
from PyPDF2 import PdfReader
from typing import List, Dict, Optional
import re
import json
import os
from tqdm import tqdm
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Import utilities from utils.py
from utils import (
    ENC,
    num_tokens,
    EMBEDDING_MODEL,
    VECTORDB_DIR,
    MANIFEST_PATH,
    calculate_file_hash,
    load_manifest,
    save_manifest,
    _cleanup_cache_files,
)


# --- Document Loading and Splitting ---
class DocLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.logger = logging.getLogger('DocLoader')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger.info(f"Initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def load_and_split(self, path: str) -> List[str]:
        self.logger.info(f"Loading and splitting PDF: {path}")
        if not os.path.exists(path):
            self.logger.error(f"PDF file not found at path: {path}")
            raise FileNotFoundError(f"PDF file not found at path: {path}")
        if not path.lower().endswith('.pdf'):
            self.logger.warning(f"File path {path} does not look like a PDF.")

        try:
            reader = PdfReader(path)
            num_pages = len(reader.pages)
            self.logger.info(f"PDF '{path}' loaded, found {num_pages} pages.")

            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

            if not full_text.strip():
                 self.logger.warning(f"No text extracted from PDF '{path}'.")
                 return []

            tokens = ENC.encode(full_text)

            chunks = []
            start = 0
            effective_overlap = min(self.chunk_overlap, self.chunk_size -1 ) if self.chunk_size > 1 else 0

            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk = ENC.decode(chunk_tokens)
                chunks.append(chunk)

                next_start = start + self.chunk_size - effective_overlap
                if len(chunk_tokens) > effective_overlap and next_start <= start :
                     next_start = start + len(chunk_tokens) - effective_overlap
                elif len(chunk_tokens) <= effective_overlap:
                     next_start = start + len(chunk_tokens)

                start = next_start

            self.logger.info(f"Finished splitting PDF '{path}' into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to load or split PDF {path}: {str(e)}", exc_info=True)
            raise


# --- Document Embedding ---
class DocEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 50):
        self.logger = logging.getLogger('DocEmbedder')
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger.info(f"Initializing DocEmbedder model component with model {self.model_name}, batch_size={batch_size}")
        self.dim: Optional[int] = None
        self.index: Optional[faiss.IndexFlatL2] = None

        try:
            # Use the specified EMBEDDING_MODEL
            dummy_embedding_response = ollama.embeddings(model=self.model_name, prompt="test")
            if not dummy_embedding_response or 'embedding' not in dummy_embedding_response:
                 raise ValueError("Ollama did not return a valid embedding for test.")
            dummy_embedding = dummy_embedding_response["embedding"]
            self.dim = len(dummy_embedding)
            self.logger.info(f"Embedding model '{self.model_name}' dimension is {self.dim}.")

        except Exception as e:
            self.logger.critical(f"Failed to initialize DocEmbedder or get model dimension: {str(e)}", exc_info=True)
            raise


    def embed(self, texts: List[str]) -> List[List[float]]:
        self.logger.info(f"Generating embeddings for {len(texts)} texts using model '{self.model_name}'.")
        if not texts:
            self.logger.warning("Empty list of texts provided for embedding.")
            return []
        if self.dim is None:
             self.logger.critical("DocEmbedder dimension not initialized.")
             raise RuntimeError("DocEmbedder not initialized properly.")

        try:
            embeddings = []
            effective_batch_size = self.batch_size
            num_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size

            # Using tqdm for console progress
            for i in tqdm(range(0, len(texts), effective_batch_size), total=num_batches, desc="Embedding texts"):
                batch = texts[i:i + effective_batch_size]

                try:
                    response = ollama.embed(model=self.model_name, input=batch)
                    batch_embeddings = response.get("embeddings")

                    if not batch_embeddings or not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(batch):
                        self.logger.warning(f"Invalid or empty embeddings for batch {i//effective_batch_size + 1}.")
                        raise ValueError(f"Ollama returned invalid or empty embeddings for batch {i//effective_batch_size + 1}")

                    if batch_embeddings and len(batch_embeddings[0]) != self.dim:
                         self.logger.error(f"Dimension mismatch for batch {i//effective_batch_size + 1}.")
                         raise ValueError(f"Dimension mismatch for batch {i//effective_batch_size + 1}")

                    embeddings.extend(batch_embeddings)
                    self.logger.debug(f"Processed batch {i//effective_batch_size + 1} with {len(batch_embeddings)} embeddings.")

                except Exception as batch_e:
                    self.logger.error(f"Error processing embedding batch {i//effective_batch_size + 1}: {batch_e}", exc_info=True)
                    raise


            self.logger.info(f"Completed embedding generation for {len(embeddings)} texts.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error during embedding process: {str(e)}", exc_info=True)
            raise


    def embed_query(self, text: str) -> List[float]:
        self.logger.debug(f"Embedding query: {text[:50]}...")
        if self.dim is None:
             self.logger.critical("DocEmbedder dimension not initialized.")
             raise RuntimeError("DocEmbedder dimension not initialized properly.")
        try:
            # Use the specified EMBEDDING_MODEL
            response = ollama.embed(model=self.model_name, input=[text])
            embedding = response.get("embeddings", [None])[0]
            if embedding is None or len(embedding) != self.dim:
                 self.logger.error(f"Invalid or empty embedding returned for query. Expected dim {self.dim}, got {len(embedding) if isinstance(embedding, list) else 'N/A'}.")
                 raise ValueError("Ollama returned invalid or empty embedding for query.")

            self.logger.debug("Query embedding generated.")
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding query: {e}", exc_info=True)
            raise


    def add_to_index(self, texts: List[str]):
        self.logger.info(f"Adding {len(texts)} embeddings to FAISS index.")
        if self.index is None:
             self.logger.critical("FAISS index is not initialized. Cannot add embeddings.")
             raise RuntimeError("FAISS index not initialized.")
        if not texts:
            self.logger.warning("No texts provided to add to index. Skipping.")
            return

        try:
            # Embedding is handled by self.embed internally which uses tqdm
            embs = self.embed(texts)
            vecs = np.array(embs, dtype="float32")
            if vecs.shape[1] != self.dim:
                self.logger.error(f"Embedding dimension mismatch before adding to index: expected {self.dim}, got {vecs.shape[1]}")
                raise ValueError(f"Embedding dimension mismatch before adding to index.")

            self.index.add(vecs)
            self.logger.info(f"Successfully added embeddings to index. Index size: {self.index.ntotal}")
        except Exception as e:
            self.logger.error(f"Failed to add embeddings to index: {str(e)}", exc_info=True)
            raise


    def load_index(self, path: str):
        self.logger.info(f"Attempting to load FAISS index from {path}.")
        if self.dim is None:
             self.logger.critical("DocEmbedder dimension not initialized.")
             raise RuntimeError("DocEmbedder dimension not initialized.")
        if not os.path.exists(path):
            self.logger.error(f"FAISS index file not found at {path}.")
            raise FileNotFoundError(f"FAISS index file not found at {path}.")
        try:
            index = faiss.read_index(path)
            if index.d != self.dim:
                 self.logger.error(f"Loaded index dimension ({index.d}) mismatches expected dimension ({self.dim}).")
                 raise ValueError("Loaded index dimension mismatch.")

            self.index = index
            self.logger.info(f"Successfully loaded FAISS index from {path}. Index size: {self.index.ntotal}, Dimension: {self.index.d}")
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index from {path}: {str(e)}", exc_info=True)
            self.index = None
            raise


    def save_index(self, path: str):
        self.logger.info(f"Attempting to save FAISS index to {path}.")
        if self.index is None:
            self.logger.warning("No FAISS index initialized on this instance to save. Skipping.")
            return
        try:
            faiss.write_index(self.index, path)
            self.logger.info(f"Successfully saved FAISS index to {path}. Index size: {self.index.ntotal}")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index to {path}: {str(e)}", exc_info=True)
            raise


    def create_new_index(self):
        if self.dim is None:
             self.logger.critical("DocEmbedder dimension not initialized.")
             raise RuntimeError("DocEmbedder dimension not initialized.")
        self.logger.info(f"Creating a new empty FAISS index with dimension {self.dim}.")
        try:
            self.index = faiss.IndexFlatL2(self.dim)
            self.logger.info("New FAISS index created.")
        except Exception as e:
            self.logger.critical(f"Failed to create new FAISS index: {str(e)}", exc_info=True)
            self.index = None
            raise


    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger('DocEmbedder.search')
        if self.index is None:
            logger.error("FAISS index not initialized on this instance. Cannot search.")
            return np.array([], dtype='float32'), np.array([], dtype='int64')

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Cannot search.")
            return np.array([], dtype='float32'), np.array([], dtype='int64')

        if query_embedding.ndim == 1:
             query_embedding = np.array([query_embedding], dtype='float32')

        if self.dim is None:
             logger.critical("DocEmbedder dimension not initialized.")
             raise RuntimeError("DocEmbedder dimension not initialized.")

        if query_embedding.shape[1] != self.dim:
             logger.error(f"Query embedding dimension mismatch: expected {self.dim}, got {query_embedding.shape[1]}")
             raise ValueError(f"Query embedding dimension mismatch.")

        k = min(k, self.index.ntotal)
        if k <= 0:
             logger.warning(f"Effective k is {k}, returning empty results.")
             return np.array([], dtype='float32'), np.array([], dtype='int64')

        logger.debug(f"Searching FAISS index ({self.index.ntotal} vectors) for {k} nearest neighbors.")
        try:
            D, I = self.index.search(query_embedding, k)
            valid_indices = [idx for idx in I[0] if idx != -1]
            valid_distances = [D[0][i] for i, idx in enumerate(I[0]) if idx != -1]

            return np.array(valid_distances, dtype='float32'), np.array(valid_indices, dtype='int64')
        except Exception as e:
            logger.error(f"Error during FAISS index search: {str(e)}", exc_info=True)
            raise


# --- Document Retrieval ---
class DocRetriever:
    def __init__(self, embedding_agent: DocEmbedder, loaded_sources: Dict[str, Dict]):
        self.logger = logging.getLogger('DocRetriever')
        self.embedding_agent = embedding_agent
        self.loaded_sources = loaded_sources
        num_sources = len(loaded_sources)
        total_chunks = sum(len(source_info.get('texts', [])) for source_info in loaded_sources.values())
        self.logger.info(f"Initialized DocRetriever with {num_sources} sources ({total_chunks} total chunks).")


    def retrieve_candidates(self, query: str, n_candidates: int = 5, k: int = 9) -> List[List[str]]:
        self.logger.info(f"Retrieving {n_candidates} candidate sets (top {k} per source) for query: {query[:100]}...")
        if not self.loaded_sources:
             self.logger.warning("No loaded sources available in DocRetriever. Cannot retrieve candidates.")
             return []

        try:
            base_emb = self.embedding_agent.embed_query(query)
        except Exception as e:
            self.logger.error(f"Failed to embed query for retrieval: {e}", exc_info=True)
            raise

        all_candidate_sets: List[List[str]] = []

        query_embeddings: List[np.ndarray] = []
        for i in range(n_candidates):
             if i == 0:
                 query_embeddings.append(np.array(base_emb, dtype='float32'))
             else:
                 perturbed_emb = np.array(base_emb, dtype='float32') + np.random.normal(0, 0.01, len(base_emb)).astype('float32')
                 query_embeddings.append(perturbed_emb)


        for i, current_query_embedding in enumerate(query_embeddings):
            current_set_chunks: List[str] = []

            for pdf_path, source_info in self.loaded_sources.items():
                 source_embedder = source_info.get('embedder')
                 source_texts = source_info.get('texts')

                 if source_embedder and source_texts and source_embedder.index is not None and source_embedder.index.ntotal > 0 and k > 0:
                     try:
                         D, I = source_embedder.search(np.array([current_query_embedding], dtype='float32'), k=k)
                         retrieved_chunks = [source_texts[j] for j in I]
                         current_set_chunks.extend(retrieved_chunks)

                     except Exception as source_search_e:
                          self.logger.error(f"Error searching source '{os.path.basename(pdf_path)}': {source_search_e}", exc_info=True)


            if current_set_chunks:
                 all_candidate_sets.append(current_set_chunks)
            else:
                 self.logger.warning(f"Query embedding {i+1} returned no chunks from any source.")


        self.logger.info(f"Finished retrieving {len(all_candidate_sets)} non-empty candidate sets.")
        return all_candidate_sets

# Note: _cleanup_cache_files is defined in utils.py