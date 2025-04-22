# knowledge_mcp/rag_manager.py
"""Manages LightRAG instances for different knowledge bases."""

import logging
from pathlib import Path

# Assuming LightRAG is installed and follows this structure
# Adjust imports based on actual LightRAG library structure if different
from lightrag import LightRAG
# TODO: Import specific model functions/clients dynamically or adjust mapping based on refined needs
# Example placeholder imports, assuming they exist directly in lightrag
from lightrag.llm.openai import openai_complete, openai_embed

from knowledge_mcp.config import Config
from lightrag.kg.shared_storage import initialize_pipeline_status

logger = logging.getLogger(__name__)

# Mapping from config provider strings to LightRAG functions/clients
# Adjust keys/values based on exact provider names in config and LightRAG functions
# TODO: Expand this mapping as needed for other providers
# --- Temporarily bypassing dynamic maps based on user feedback ---
# EMBED_FUNC_MAP = {
#     "openai": openai_embed,
#     # "cohere": cohere_embed, # Example
# }
#
# LLM_FUNC_MAP = {
#     "openai": openai_complete,
#     # "cohere": cohere_complete, # Example
# }
#
# # Mapping for keyword arguments expected by LightRAG functions based on provider
# MODEL_KWARGS_MAP = {
#     "openai": {"api_key": "api_key", "base_url": "base_url"},
#     # "cohere": {"api_key": "api_key"}, # Example
# }


class RAGManagerError(Exception):
    """Base exception for RAGManager errors."""

class UnsupportedProviderError(RAGManagerError):
    """Raised when a configured provider is not supported."""

class RAGInitializationError(RAGManagerError):
    """Raised when LightRAG instance initialization fails."""


class RAGManager:
    """Creates, manages, and caches LightRAG instances per knowledge base."""

    def __init__(self, config: Config):
        """Initializes the RAGManager with application config."""
        self.config = config
        self._rag_instances: dict[str, LightRAG] = {}
        self._kb_base_dir = Path(config.knowledge_base.base_dir) # Ensure it's a Path
        logger.info(f"RAGManager initialized with KB base directory: {self._kb_base_dir}")

    async def get_rag_instance(self, kb_name: str) -> LightRAG:
        """
        Retrieves or creates and initializes a LightRAG instance for the given KB.

        Args:
            kb_name: The name of the knowledge base.

        Returns:
            An initialized LightRAG instance.

        Raises:
            KnowledgeBaseNotFoundError: If the underlying KB directory doesn't exist (optional check).
            UnsupportedProviderError: If embedding or LLM provider is not supported.
            RAGInitializationError: If LightRAG fails to initialize.
            ValueError: If required configuration (e.g., API key) is missing.
        """
        if kb_name in self._rag_instances:
            logger.debug(f"Returning cached LightRAG instance for KB: {kb_name}")
            return self._rag_instances[kb_name]

        logger.info(f"Creating new LightRAG instance for KB: {kb_name}")
        kb_path = self._kb_base_dir / kb_name

        # Optional: Check if the KB directory exists via KnowledgeBaseManager
        # kb_manager = KnowledgeBaseManager(self._kb_base_dir) # Or pass manager instance
        # if not kb_manager.kb_exists(kb_name): # Assuming kb_exists method
        #     raise KnowledgeBaseNotFoundError(f"Knowledge base directory does not exist: {kb_path}")
        # Ensure the directory exists for LightRAG working_dir
        if not kb_path.is_dir():
             # If we rely on KB Manager creating it, this might indicate an issue.
             # Or, RAGManager could create it, but KBManager seems more appropriate.
             # For now, assume KBManager ensures the dir exists before processing.
             logger.warning(f"Knowledge base directory {kb_path} not found. Assuming it will be created.")
             # Alternatively, raise error here if RAG init requires existing dir.

        try:
            llm_config = self.config.language_model # Needed for LightRAG init

            # --- Get Embedding Function and Kwargs --- Refactored ---
            # Directly use openai_embed, assuming API key is handled via env or constructor needs it
            # embed_provider = embed_config.provider.lower()
            # if embed_provider != "openai": # Simple check for now
            #     raise UnsupportedProviderError(f"Only OpenAI embedding provider currently supported in this refactor.")

            # Construct embedding kwargs if needed by openai_embed directly
            # embed_kwargs = {}
            # api_key = getattr(embed_config, "api_key", None)
            # base_url = getattr(embed_config, "base_url", None)
            # if not api_key:
            #      raise ValueError("API key missing for OpenAI embedding provider")
            # embed_kwargs["api_key"] = api_key
            # if base_url:
            #      embed_kwargs["base_url"] = base_url

            # --- Get LLM Function and Kwargs --- Refactored ---
            # Directly use openai_complete
            # llm_provider = llm_config.provider.lower()
            # if llm_provider != "openai": # Simple check for now
            #      raise UnsupportedProviderError(f"Only OpenAI language model provider currently supported in this refactor.")

            llm_kwargs = {
                "api_key": llm_config.api_key,
                "model": llm_config.model_name,
                # Add other potential kwargs like base_url if needed
            }
            if llm_config.base_url:
                llm_kwargs["base_url"] = llm_config.base_url
            # Add max_tokens if present in config - USE THE CORRECT ARGUMENT NAME for LightRAG
            llm_model_max_tokens = getattr(llm_config, 'max_tokens', None)
            print("llm_model_max_tokens", llm_model_max_tokens)
            logger.debug(
                f"Attempting to initialize LightRAG for {kb_name} with parameters:\n"
                f"  working_dir: {kb_path}\n"
                f"  llm_model_kwargs: {llm_kwargs}\n"
                f"  llm_model_name: {llm_config.model_name}\n"
                f"  llm_model_max_token_size: {llm_model_max_tokens}"
            )

            # --- Instantiate LightRAG --- Refactored ---
            logger.debug(f"Instantiating LightRAG for {kb_name} with working_dir: {kb_path}")
            rag = LightRAG(
                working_dir=str(kb_path),
                embedding_func=openai_embed, # Directly use imported function
                # embedding_model_kwargs=embed_kwargs, # Pass if needed by openai_embed signature
                llm_model_func=openai_complete, # Directly use imported function
                llm_model_kwargs=llm_kwargs, # Pass constructed kwargs
                llm_model_name=llm_config.model_name, # Get model name from config
                llm_model_max_token_size=llm_model_max_tokens # Pass max_tokens from config
            )

            # --- Initialize Storages ---
            logger.debug(f"Initializing LightRAG storages for {kb_name}...")
            await rag.initialize_storages()
            await initialize_pipeline_status()
            logger.info(f"Successfully initialized LightRAG instance for KB: {kb_name}")

            self._rag_instances[kb_name] = rag
            return rag

        except (UnsupportedProviderError, ValueError) as e:
             logger.error(f"Configuration error for KB '{kb_name}': {e}")
             raise # Re-raise config errors
        except Exception as e:
            # Catch potential errors during LightRAG init or storage init
            logger.exception(f"Failed to initialize LightRAG for KB '{kb_name}' at {kb_path}: {e}")
            raise RAGInitializationError(f"LightRAG initialization failed for KB '{kb_name}': {e}") from e

    def clear_cache(self, kb_name: str | None = None) -> None:
        """Clears the cache for a specific KB or all KBs."""
        if kb_name:
            if kb_name in self._rag_instances:
                del self._rag_instances[kb_name]
                logger.info(f"Cleared cached LightRAG instance for KB: {kb_name}")
        else:
            self._rag_instances.clear()
            logger.info("Cleared all cached LightRAG instances.")
