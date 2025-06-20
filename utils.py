import logging
import os
import sys
import json
import hashlib
from datetime import datetime
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
import tiktoken
from typing import List, Dict, Optional
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --- Logging Configuration ---
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'rag.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger('Utils') # Use 'Utils' logger

# --- Model Definitions ---
MODELS_CONFIG_FILE = "models.json"

model_config = {}
try:
    if os.path.exists(MODELS_CONFIG_FILE):
        with open(MODELS_CONFIG_FILE, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        logger.info(f"Loaded model configuration from {MODELS_CONFIG_FILE}.")
    else:
        logger.critical(f"Model configuration file not found at {MODELS_CONFIG_FILE}. Please create it.")
        sys.exit(1)

except json.JSONDecodeError as e:
    logger.critical(f"Failed to parse model configuration JSON at {MODELS_CONFIG_FILE}: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"Failed to load model configuration from {MODELS_CONFIG_FILE}: {e}", exc_info=True)
    sys.exit(1)

LLM_BASE_URL = model_config.get("base_url")
LLM_MODELS = model_config.get("models", [])

if not LLM_BASE_URL:
    logger.critical(f"'base_url' not found in {MODELS_CONFIG_FILE}.")
    sys.exit(1)
if not LLM_MODELS:
    logger.critical(f"'models' list is empty or not found in {MODELS_CONFIG_FILE}.")
    sys.exit(1)

logger.info(f"Configured LLM base_url: {LLM_BASE_URL}")
logger.info(f"Available LLM models: {LLM_MODELS}")

EMBEDDING_MODEL = "nomic-embed-text"
logger.info(f"Using embedding model: {EMBEDDING_MODEL}")

# --- LLM Client and Model Manager ---
class ModelManager:
    def __init__(self, base_url: str, models: List[str]):
        self.logger = logging.getLogger('ModelManager')
        if not models:
            self.logger.critical("Model list is empty. Cannot initialize ModelManager.")
            raise ValueError("Model list cannot be empty.")
        if not base_url:
             self.logger.critical("Base URL is not provided. Cannot initialize ModelManager.")
             raise ValueError("Base URL cannot be empty.")

        self.base_url = base_url
        self.models = models
        self.current_model_index = 0
        self.client: Optional[OpenAI] = None
        self._initialize_client()
        self.logger.info(f"Initialized ModelManager with base_url={base_url} and {len(models)} models.")

    def _initialize_client(self):
        self.logger.info(f"Initializing OpenAI client with base_url: {self.base_url}")
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                self.logger.critical("GOOGLE_API_KEY environment variable not set. Cannot initialize LLM client.")
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            self.client = OpenAI(
                api_key=google_api_key,
                base_url=self.base_url
            )
            self.logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            self.client = None
            raise

    def get_client(self) -> OpenAI:
        if self.client is None:
            self.logger.critical("Attempted to get client before initialization or after failure.")
            raise RuntimeError("LLM client is not initialized.")
        return self.client

    def get_current_model_name(self) -> str:
        if not self.models:
             self.logger.critical("Model list is empty in get_current_model_name.")
             raise RuntimeError("Model list is empty.")
        if not 0 <= self.current_model_index < len(self.models):
             self.logger.error(f"Current model index {self.current_model_index} out of bounds (0-{len(self.models)-1}). Resetting to 0.")
             self.current_model_index = 0

        return self.models[self.current_model_index]

    def cycle_model(self) -> bool:
        if not self.models or len(self.models) <= 1:
            self.logger.warning("Only one or no models available. Cannot cycle.")
            return False

        initial_index = self.current_model_index
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        cycled_back_to_start = self.current_model_index == initial_index

        if cycled_back_to_start:
             self.logger.warning(f"Cycled back to the first model in the list ({self.get_current_model_name()}). No new models left to try.")
             return False

        self.logger.warning(f"Cycled to next model: {self.get_current_model_name()}.")
        return True

    def handle_api_error(self, error: Exception) -> bool:
        self.logger.error(f"API call encountered an error: {type(error).__name__}: {error}", exc_info=True)

        is_recoverable_error = False
        if isinstance(error, (APIStatusError, APIConnectionError, RateLimitError)):
            if isinstance(error, APIStatusError):
                if error.status_code in {404, 429, 500, 503}:
                    self.logger.warning(f"API Status Error {error.status_code}. Considered potentially recoverable.")
                    is_recoverable_error = True
                elif 400 <= error.status_code < 500:
                    self.logger.error(f"API Status Error {error.status_code}. Considered non-recoverable (client-side/request error).")
                    return False
                else:
                     self.logger.warning(f"Unexpected API Status Error {error.status_code}. Cycling model just in case.")
                     is_recoverable_error = True
            elif isinstance(error, (APIConnectionError, RateLimitError)):
                 self.logger.warning(f"API Connection Error or Rate Limit Error. Considered potentially recoverable.")
                 is_recoverable_error = True
            else:
                 self.logger.warning(f"Caught an API exception type that is not explicitly handled ({type(error).__name__}). Cycling model just in case.")
                 is_recoverable_error = True

        elif isinstance(error, ValueError) and "out of bounds" in str(error).lower():
             self.logger.warning("Caught ValueError related to out-of-bounds index, treating as potentially recoverable model issue.")
             is_recoverable_error = True

        if is_recoverable_error:
             if self.cycle_model():
                 self.logger.warning(f"Cycling to next model: {self.get_current_model_name()}.")
                 return True
             else:
                 self.logger.error("No more models left to cycle through.")
                 return False

        else:
            self.logger.error("API error is not considered recoverable. Not cycling model.")
            return False

try:
    model_manager = ModelManager(LLM_BASE_URL, LLM_MODELS)
except Exception as e:
    logger.critical(f"Failed to initialize ModelManager: {e}")
    sys.exit(1)


# --- Tokenizer ---
logger.info("Loading tiktoken encoding.")
try:
    ENC = tiktoken.get_encoding("cl100k_base")
    logger.info("Tiktoken encoding loaded.")
except Exception as e:
     logger.critical(f"Failed to load tiktoken encoding: {e}", exc_info=True)
     sys.exit(1)


def num_tokens(text: str) -> int:
    return len(ENC.encode(text))


# --- Persistence Configuration ---
VECTORDB_DIR = "vectordb"
MANIFEST_FILE = "manifest.json"
MANIFEST_PATH = os.path.join(VECTORDB_DIR, MANIFEST_FILE)
os.makedirs(VECTORDB_DIR, exist_ok=True)
logger = logging.getLogger('Global') # Re-get logger after basicConfig potentially
logger.info(f"Vector database cache directory: {VECTORDB_DIR}")


# --- File Hashing ---
def calculate_file_hash(file_path: str, algorithm: str = 'sha256', chunk_size: int = 4096) -> str:
    logger = logging.getLogger('FileHash')
    try:
        hash_func = hashlib.sha256 if algorithm == 'sha256' else hashlib.md5
        hasher = hash_func()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}", exc_info=True)
        raise


# --- Manifest Management ---
def load_manifest(manifest_path: str) -> Dict:
    logger = logging.getLogger('Manifest')
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to load manifest from {manifest_path}: {e}", exc_info=True)
        return {}


def save_manifest(manifest_path: str, manifest: Dict):
    logger = logging.getLogger('Manifest')
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save manifest to {manifest_path}: {e}", exc_info=True)


# --- Cache Cleanup Helper ---
def _cleanup_cache_files(index_path: str | None, texts_path: str | None):
    logger = logging.getLogger('CleanupCache')
    if index_path and os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception as e:
            logger.warning(f"Failed to delete old index file {index_path}: {e}", exc_info=True)
    if texts_path and os.path.exists(texts_path):
        try:
            os.remove(texts_path)
        except Exception as e:
            logger.warning(f"Failed to delete old texts file {texts_path}: {e}", exc_info=True)