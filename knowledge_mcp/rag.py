# knowledge_mcp/rag_manager.py
"""Manages LightRAG instances for different knowledge bases."""

import logging
import logging.handlers
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from typing import Dict, Optional, Any
import asyncio

# Need to import Config and KbManager to use them
from knowledge_mcp.config import Config
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError

logger = logging.getLogger(__name__) # General logger for RagManager setup/errors not specific to a KB

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
        self._rag_instances: Dict[str, LightRAG] = {}
        self._kb_loggers: Dict[str, logging.Logger] = {}
        self.kb_manager = kb_manager 
        logger.info("RagManager initialized.") 

    def _get_kb_logger(self, kb_name: str) -> logging.Logger:
        """Gets or creates a logger specific to a knowledge base."""
        logger_name = f"kbmcp.{kb_name}"
        if logger_name in self._kb_loggers:
            return self._kb_loggers[logger_name]

        # --- Configure new KB logger --- 
        kb_logger = logging.getLogger(logger_name)
        
        # Check if handlers are already configured (e.g., by a different process/thread? unlikely here but safe)
        if not kb_logger.handlers:
            main_log_config = Config.get_instance().logging
            try:
                kb_path = self.kb_manager.get_kb_path(kb_name)
            except KnowledgeBaseNotFoundError:
                 # Should not happen if called after KB exists, but handle defensively
                 logger.error(f"Attempted to get logger for non-existent KB '{kb_name}'")
                 # Return the main logger as a fallback? Or raise error?
                 # Returning main logger might hide issues. Let's return the unconfigured kb_logger.
                 return kb_logger
            
            kb_log_dir = kb_path / "logs"
            kb_log_file = kb_log_dir / f"kbmcp-{kb_name}.log"
            kb_log_dir.mkdir(parents=True, exist_ok=True)

            # File Handler
            file_formatter = logging.Formatter(main_log_config.detailed_format)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=kb_log_file,
                maxBytes=main_log_config.max_bytes,
                backupCount=main_log_config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(main_log_config.level)
            kb_logger.addHandler(file_handler)

            # Console Handler (optional, could let main logger handle console)
            # If added, KB-specific messages will appear twice on console (once via main, once via KB logger)
            # Let's skip adding a separate console handler here to avoid duplicate console logs.
            # console_formatter = logging.Formatter(main_log_config.default_format)
            # console_handler = logging.StreamHandler()
            # console_handler.setFormatter(console_formatter)
            # console_handler.setLevel(main_log_config.level)
            # kb_logger.addHandler(console_handler)

            kb_logger.setLevel(main_log_config.level)
            kb_logger.propagate = False # Crucial: Prevent messages going to root/kbmcp handler
            logger.info(f"Configured logger '{logger_name}' to file: {kb_log_file}")
            
        self._kb_loggers[logger_name] = kb_logger
        return kb_logger

    def get_rag_instance(self, kb_name: str) -> LightRAG:
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
            self._get_kb_logger(kb_name).info("Returning cached LightRAG instance.")
            return self._rag_instances[kb_name]
        else:
            if self.kb_manager.kb_exists(kb_name):
                return asyncio.run(self.create_rag_instance(kb_name))
            else:
                raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist.")              
    
    async def create_rag_instance(self, kb_name: str) -> LightRAG:
        # Use KbManager to check existence and get path
        if not self.kb_manager.kb_exists(kb_name):
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist.")
        kb_path = self.kb_manager.get_kb_path(kb_name)
        kb_logger = self._get_kb_logger(kb_name)
        kb_logger.info(f"Creating new LightRAG instance in {kb_path}")

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

            kb_logger.info(
                "Attempting to initialize LightRAG with parameters:\n"
                f"  working_dir: {kb_path}\n"
                f"  embed_model: {embed_config.model_name}\n"
                f"  llm_model: {llm_config.model_name}, llm_kwargs: {llm_kwargs}\n"
                f"  llm_model_max_token_size: {llm_model_max_tokens}"
            )

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

            # --- Initialize Storages/Components ---
            kb_logger.debug(f"Initializing LightRAG components for {kb_name}...")
            # Check LightRAG documentation for the correct initialization method
            # It might be initialize_components(), initialize_storages(), or similar.
            # Assuming initialize_components() based on common patterns
            await rag.initialize_storages()
            await initialize_pipeline_status()
            kb_logger.info("Successfully initialized LightRAG instance.")

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

    async def ingest_document(self, kb_name: str, file_path: Any, doc_id: Optional[str] = None) -> Optional[str]:
        """Ingests a document into the specified knowledge base."""
        kb_logger = self._get_kb_logger(kb_name)
        kb_logger.info(f"Ingesting document '{file_path.name}' into KB '{kb_name}'...")
        
        try:
            rag_instance = await self.get_rag_instance(kb_name)
            
            # Use LightRAG's ingest_doc method
            # Check LightRAG docs for exact parameters and return value
            # Assuming it takes file path and optional doc_id
            # result = await rag_instance.ingest_doc(doc_path=str(file_path), doc_id=doc_id)
            # Simplified call for now, assuming file path is enough
            await rag_instance.ingest_doc(doc_path=str(file_path))
            
            # Assuming ingest_doc doesn't directly return a useful ID in this version
            # We might need to generate/manage IDs separately if required.
            generated_doc_id = doc_id or file_path.stem # Placeholder ID logic
            kb_logger.info(f"Successfully ingested document '{file_path.name}' (ID: {generated_doc_id}).")
            return generated_doc_id
        
        except RAGInitializationError as e:
            kb_logger.error(f"Cannot ingest, RAG instance failed to initialize: {e}")
            raise # Re-raise the initialization error
        except FileNotFoundError:
            kb_logger.error(f"Document file not found: {file_path}")
            raise
        except Exception as e:
            kb_logger.exception(f"Failed to ingest document '{file_path.name}': {e}")
            # Consider wrapping in a specific IngestionError if needed
            raise RAGManagerError(f"Ingestion failed for '{file_path.name}': {e}") from e

    def query(self, kb_name: str, query_text: str, **kwargs: Any) -> Any:
        """Performs a query against the specified knowledge base."""
        kb_logger = self._get_kb_logger(kb_name)
        
        try:
            rag_instance = self.get_rag_instance(kb_name)
            
            # Use LightRAG's query method
            # Check LightRAG docs for parameters and return structure
            kb_logger.debug(f"Forwarding query to LightRAG instance with kwargs: {kwargs}")
            result = rag_instance.query(query=query_text, **kwargs)
            kb_logger.debug(f"Query result: {result}") # Be mindful of logging sensitive data
            return result
        
        except RAGInitializationError as e:
            kb_logger.error(f"Cannot query, RAG instance failed to initialize: {e}")
            raise # Re-raise the initialization error
        except Exception as e:
            kb_logger.exception(f"Failed to process query '{query_text}': {e}")
            raise RAGManagerError(f"Query failed: {e}") from e
