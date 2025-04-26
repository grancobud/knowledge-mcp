# knowledge_mcp/rag_manager.py
"""Manages LightRAG instances for different knowledge bases."""

import logging
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

# Need to import Config and KbManager to use them
from knowledge_mcp.config import Config
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError

logger = logging.getLogger(__name__)

# Removed provider maps as they are not currently used directly


class RAGManagerError(Exception):
    """Base exception for RAGManager errors."""

class UnsupportedProviderError(RAGManagerError):
    """Raised when a configured provider is not supported."""

class RAGInitializationError(RAGManagerError):
    """Raised when LightRAG instance initialization fails."""

class ConfigurationError(RAGManagerError):
    """Raised when required configuration is missing or invalid."""


class RagManager:
    """Creates, manages, and caches LightRAG instances per knowledge base."""

    def __init__(self, config: Config, kb_manager: KnowledgeBaseManager): 
        """Initializes the RagManager with the KB manager."""
        self._rag_instances: dict[str, LightRAG] = {}
        self.kb_manager = kb_manager 
        logger.info("RagManager initialized.") 

    async def get_rag_instance(self, kb_name: str) -> LightRAG:
        """
        Retrieves or creates and initializes a LightRAG instance for the given KB.

        Args:
            kb_name: The name of the knowledge base.

        Returns:
            An initialized LightRAG instance.

        Raises:
            KnowledgeBaseNotFoundError: If the underlying KB directory doesn't exist.
            UnsupportedProviderError: If embedding or LLM provider is not supported (currently only checks OpenAI).
            ConfigurationError: If required configuration (e.g., API key) is missing.
            RAGInitializationError: If LightRAG fails to initialize.
        """
        if kb_name in self._rag_instances:
            logger.debug(f"Returning cached LightRAG instance for KB: {kb_name}")
            return self._rag_instances[kb_name]

        # Use KbManager to check existence and get path
        if not self.kb_manager.kb_exists(kb_name):
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist.")
        kb_path = self.kb_manager.get_kb_path(kb_name)
        logger.info(f"Creating new LightRAG instance for KB: {kb_name} in {kb_path}")

        try:
            # Get the singleton config instance
            config = Config.get_instance()

            # Validate required settings sections exist
            if not config.lightrag or not config.lightrag.llm:
                 raise ConfigurationError("Language model settings (config.lightrag.llm) are missing.")
            if not config.lightrag.embedding:
                 raise ConfigurationError("Embedding model settings (config.lightrag.embedding) are missing.")
            if not config.lightrag.embedding_cache:
                 raise ConfigurationError("Embedding cache settings (config.lightrag.embedding_cache) are missing.")

            llm_config = config.lightrag.llm
            embed_config = config.lightrag.embedding
            cache_config = config.lightrag.embedding_cache

            # --- Get Embedding Function and Kwargs ---
            embed_provider = embed_config.provider.lower()
            if embed_provider == "openai":
                import knowledge_mcp.openai_func
                embed_func = knowledge_mcp.openai_func.embedding_func
            else:
                raise UnsupportedProviderError("Only OpenAI embedding provider currently supported.") 

            # --- Get LLM Function and Kwargs ---
            llm_provider = llm_config.provider.lower()
            if llm_provider == "openai":
                import knowledge_mcp.openai_func
                llm_func = knowledge_mcp.openai_func.llm_model_func
            else:
                raise UnsupportedProviderError("Only OpenAI language model provider currently supported.") 

            if not llm_config.api_key:
                 raise ConfigurationError("API key missing for OpenAI language model provider")
            # llm_kwargs={"api_key": llm_config.api_key}
            # if llm_config.api_base:
            #     llm_kwargs["base_url"] = llm_config.api_base
            # Add other potential kwargs if needed (e.g., temperature, etc.)
            llm_kwargs = {}
            if llm_config.kwargs:
                llm_kwargs.update(llm_config.kwargs)

            # Max tokens for the LLM *model* (for context window sizing)
            # Use LightRAG default if not set, check LightRAG docs for correct handling
            llm_model_max_tokens = llm_config.max_token_size

            logger.info(
                f"Attempting to initialize LightRAG for {kb_name} with parameters:\n"
                f"  working_dir: {kb_path}\n"
                f"  embed_model: {embed_config.model_name}\n"
                f"  llm_model: {llm_config.model_name}, llm_kwargs: {llm_kwargs}\n"
                f"  llm_model_max_token_size: {llm_model_max_tokens}"
            )
            print(kb_path)
            # --- Instantiate LightRAG ---
            # Note: Verify LightRAG constructor parameters closely with LightRAG docs
            rag = LightRAG(
                working_dir=str(kb_path),
                llm_model_func=llm_func,
                llm_model_kwargs=llm_kwargs,
                llm_model_name=llm_config.model_name, 
                llm_model_max_token_size=llm_model_max_tokens,
                embedding_func=embed_func,
                embedding_cache_config={
                    "enabled": cache_config.enabled,
                    "similarity_threshold": cache_config.similarity_threshold,
                },              
            )
            print(rag)
            # --- Initialize Storages/Components ---
            logger.debug(f"Initializing LightRAG components for {kb_name}...")
            # Check LightRAG documentation for the correct initialization method
            # It might be initialize_components(), initialize_storages(), or similar.
            # Assuming initialize_components() based on common patterns
            await rag.initialize_storages()
            await initialize_pipeline_status()
            logger.info(f"Successfully initialized LightRAG instance for KB: {kb_name}") 

            self._rag_instances[kb_name] = rag
            return rag

        except (UnsupportedProviderError, ConfigurationError, KnowledgeBaseNotFoundError) as e:
             logger.error(f"Setup error for KB '{kb_name}': {e}")
             raise 
        except Exception as e:
            # Catch potential errors during LightRAG init or storage init
            logger.exception(f"Failed to initialize LightRAG for KB '{kb_name}' at {kb_path}: {e}")
            raise RAGInitializationError(f"LightRAG initialization failed for KB '{kb_name}': {e}") from e

    # --- Search Method (Placeholder) ---
    async def search(self, kb_name: str, query: str):
        # Placeholder - Implementation needed
        logger.warning(f"search not fully implemented. Args: {kb_name}, {query}")
        pass

    def remove_rag_instance(self, kb_name: str | None = None) -> None:
        """Removes a rag instance by name"""
        if kb_name:
            if kb_name in self._rag_instances:
                del self._rag_instances[kb_name]
                logger.info(f"Removed LightRAG instance for KB: {kb_name}")
            else:
                logger.error(f"Knowledge base '{kb_name}' not found.")
                raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' not found.")
        else:
            logger.error("Knowledgebase name is required.")
            raise ValueError("Knowledgebase name is required.")
