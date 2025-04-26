"""
Configuration parsing with YAML support, environment variable substitution,
and singleton access via the Config class itself.
"""
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class KnowledgeBaseConfig(BaseModel):
    """Configuration for the knowledge base storage."""
    base_dir: Path = Field(..., description="Base directory for knowledge base storage")

class LightRAGEmbeddingConfig(BaseModel):
    """Configuration specific to the LightRAG embedding model setup."""
    provider: str = Field(..., description="Provider name (e.g., 'openai')")
    model_name: str = Field(..., description="Name of the embedding model")
    api_key: str = Field(..., description="API key, supports env var substitution like ${VAR_NAME}")
    api_base: Optional[str] = Field(None, description="Optional base URL for the API, supports env var substitution")
    embedding_dim: int = Field(..., description="Dimensionality of the embeddings")
    max_token_size: int = Field(..., description="Maximum token size supported by the embedding model")

class LightRAGLLMConfig(BaseModel):
    """Configuration specific to the LightRAG language model setup."""
    provider: str = Field(..., description="Provider name (e.g., 'openai')")
    model_name: str = Field(..., description="Name of the language model")
    max_token_size: int = Field(..., description="Maximum token size for the model context")
    api_key: str = Field(..., description="API key, supports env var substitution like ${VAR_NAME}")
    api_base: Optional[str] = Field(None, description="Optional base URL for the API, supports env var substitution")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Dynamic keyword arguments for the LLM provider")

class EmbeddingCacheConfig(BaseModel):
    """Configuration for the embedding cache."""
    enabled: bool = Field(..., description="Whether the embedding cache is enabled")
    similarity_threshold: float = Field(..., description="Similarity threshold for cache hits")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level (e.g., DEBUG, INFO, WARNING)")
    max_bytes: int = Field(default=10485760, description="Max size of log file in bytes before rotation (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files to keep")
    detailed_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Format string for detailed logs")
    default_format: str = Field(default="%(levelname)s: %(message)s", description="Format string for console logs")

class LightRAGConfig(BaseModel):
    """Configuration specific to the LightRAG library components."""
    llm: LightRAGLLMConfig
    embedding: LightRAGEmbeddingConfig
    embedding_cache: EmbeddingCacheConfig

# Type variable for the Config class itself
C = TypeVar('C', bound='Config')

# Define the lock at the module level, associated with the Config class logic
_config_load_lock = threading.Lock()

class Config(BaseModel):
    """
    Root configuration model. Manages its own singleton instance.
    Load configuration data using Config.load(path) before calling Config.get_instance().
    """
    knowledge_base: KnowledgeBaseConfig
    lightrag: LightRAGConfig
    logging: LoggingConfig
    env_file: Path = Field(..., description="Path to the .env file relative to the config file")

    # --- Singleton Management ---
    _instance: Optional['Config'] = None
    _loaded: bool = False

    @staticmethod
    def _resolve_path(base_path: Path, relative_path: Path | str) -> Path:
        """Resolves a path relative to a base directory path."""
        return (base_path.parent / relative_path).resolve()

    @staticmethod
    def _read_and_process_config(config_path: str | Path) -> Dict[str, Any]:
        """Reads, substitutes env vars, parses YAML, resolves paths."""
        config_file = Path(config_path).resolve()
        logger.info(f"Reading configuration file: {config_file}")

        if not config_file.is_file():
            logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # 1. Read raw content
        try:
            raw_content = config_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.exception(f"Error reading config file {config_file}")
            raise RuntimeError(f"Could not read config file {config_file}: {e}") from e

        # 2. Preliminary parse to find env_file path
        env_file_path_relative: Optional[str] = None
        try:
            prelim_dict = yaml.safe_load(raw_content)
            if isinstance(prelim_dict, dict):
                env_file_path_relative = prelim_dict.get("env_file")
            else:
                logger.warning(f"Config file {config_file} root is not a dictionary.")
        except yaml.YAMLError as e:
            logger.warning(f"Could not preliminary parse YAML in {config_file}: {e}")

        # 3. Load .env file if specified
        if env_file_path_relative:
            env_file_path = Config._resolve_path(config_file, env_file_path_relative)
            logger.info(f"Resolved env_file path: {env_file_path}")
            if env_file_path.is_file():
                logger.info(f"Loading environment variables from: {env_file_path}")
                try:
                    load_dotenv(dotenv_path=env_file_path, override=True)
                except Exception as e:
                    logger.exception(f"Error loading .env file: {env_file_path}")
                    logger.warning(f"Proceeding despite error loading .env file: {e}")
            else:
                logger.error(f"Specified env_file not found: {env_file_path}")
                raise FileNotFoundError(f"Specified env_file not found: {env_file_path}")
        else:
            logger.info("No 'env_file' specified. Skipping .env loading.")

        # 4. Substitute environment variables ${VAR_NAME}
        def replace_env_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning(f"Env var '{var_name}' not found for substitution.")
                return match.group(0)
            return value

        try:
            substituted_content = re.sub(r"\$\{(\w+)\}", replace_env_var, raw_content)
        except Exception as e:
            logger.exception("Error during env var substitution.")
            raise RuntimeError("Failed during env var substitution.") from e

        # 5. Parse final YAML
        try:
            config_dict = yaml.safe_load(substituted_content)
            if not isinstance(config_dict, dict):
                raise ValueError("YAML root is not a dictionary.")

            # Resolve paths within the dictionary before validation
            if "knowledge_base" in config_dict and "base_dir" in config_dict["knowledge_base"]:
                kb_base_dir = config_dict["knowledge_base"]["base_dir"]
                config_dict["knowledge_base"]["base_dir"] = Config._resolve_path(config_file, kb_base_dir)
            if "env_file" in config_dict:
                 config_dict["env_file"] = Config._resolve_path(config_file, config_dict["env_file"])

            return config_dict

        except yaml.YAMLError as e:
            logger.exception(f"Error parsing final YAML: {config_file}")
            raise ValueError(f"Invalid YAML format: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error processing config dict: {config_file}")
            raise RuntimeError(f"Failed to process config dict: {e}") from e

    @classmethod
    def load(cls: Type[C], config_path: str | Path = "config.yaml") -> None:
        """
        Loads configuration from the specified path, validates it,
        and stores it as the singleton instance. Should only be called once.

        Args:
            config_path: Path to the configuration YAML file.
        """
        try:
            config_dict = cls._read_and_process_config(config_path)
            # Validate and create the instance using Pydantic's method
            cls._instance = cls(**config_dict)
            cls._loaded = True
            logger.info("Configuration loaded and validated successfully.")
        except (FileNotFoundError, ValueError, ValidationError, RuntimeError) as e:
            logger.error(f"Failed to load or validate configuration from {config_path}: {e}", exc_info=True)
            # Clear potentially partially loaded state
            cls._instance = None
            cls._loaded = False
            raise RuntimeError(f"Configuration loading failed. Original error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during configuration load from {config_path}: {e}", exc_info=True)
            cls._instance = None
            cls._loaded = False
            raise RuntimeError(f"Unexpected configuration loading failed. Original error: {e}") from e


    @classmethod
    def get_instance(cls: Type[C]) -> C:
        """
        Returns the loaded singleton configuration instance.

        Raises:
            RuntimeError: If [load()] has not been successfully called beforehand.
        """
        if not cls._loaded or cls._instance is None:
            raise RuntimeError("Configuration has not been loaded. Call load(path) first.")
        return cls._instance