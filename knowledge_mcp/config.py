"""
Configuration parsing with YAML support and environment variable substitution.
"""
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class KnowledgeBaseConfig(BaseModel):
    base_dir: Path = Field(..., description="Base directory for knowledge base storage")


class EmbeddingModelConfig(BaseModel):
    provider: str = Field(..., description="Provider name (e.g., 'openai')")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    model_name: str = Field(..., description="Name of the embedding model")
    api_key: str = Field(..., description="API key, supports env var substitution")


class LanguageModelConfig(BaseModel):
    provider: str = Field(..., description="Provider name (e.g., 'openai')")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    model_name: str = Field(..., description="Name of the language model")
    api_key: str = Field(..., description="API key, supports env var substitution")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for the model")


class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="Logging level")
    file: Optional[Path] = Field(None, description="Path to the log file")


class Config(BaseModel):
    knowledge_base: KnowledgeBaseConfig
    embedding_model: EmbeddingModelConfig
    language_model: LanguageModelConfig
    logging: LoggingConfig


# --- Config Loading Function ---
def load_and_validate_config(config_path: str | Path) -> Config:
    """
    Loads YAML config, substitutes env vars, and validates with Pydantic.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A validated Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If YAML is invalid, validation fails, or structure is wrong.
        RuntimeError: For other unexpected errors during loading.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found or is not a file: {config_path}")

    logger.info(f"Reading config from: {config_file.resolve()}")
    raw = config_file.read_text()

    # Substitute environment variables (e.g., ${API_KEY})
    substituted_raw = raw
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        if placeholder in substituted_raw:
            logger.debug(f"Substituting {placeholder} from environment variable.")
            substituted_raw = substituted_raw.replace(placeholder, value)

    try:
        config_dict = yaml.safe_load(substituted_raw)
        if not isinstance(config_dict, dict):
            raise ValueError(f"Invalid YAML format in {config_path}: Expected a dictionary, got {type(config_dict)}")

        validated_config = Config.model_validate(config_dict)
        logger.info("Configuration loaded and validated successfully.")

        # Load environment variables from env_file
        if config_dict["env_file"]:
            env_file = Path(config_dict["env_file"])
            if not env_file.is_file():
                raise FileNotFoundError(f"Environment file not found or is not a file: {env_file}")
            logger.info(f"Loading environment variables from: {env_file.resolve()}")
            load_dotenv(dotenv_path=env_file)
            logger.info("Environment variables loaded successfully.")

        return validated_config
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML config file: {config_path}")
        raise ValueError(f"Invalid YAML format in {config_path}: {e}") from e
    except ValidationError as e:
        logger.exception(f"Configuration validation failed for {config_path}")
        error_details = '\n'.join([f"  - {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid configuration in {config_path}:\n{error_details}") from e
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading config {config_path}")
        raise RuntimeError(f"Failed to load config {config_path}: {e}") from e


# Example Usage (Optional, for testing the module directly)
if __name__ == "__main__":
    example_config_path = Path("config.yaml")
    if not example_config_path.exists():
        print("Creating dummy config.yaml for __main__ example...")
        dummy_config = {
            'knowledge_base': {'base_dir': './kb_data'},
            'embedding_model': {
                'provider': 'example_emb',
                'model_name': 'example-emb-model',
                'api_key': 'dummy_emb_key'
            },
            'language_model': {
                'provider': 'example_llm',
                'model_name': 'example-llm-model',
                'api_key': 'dummy_llm_key'
            },
            'logging': {'level': 'INFO'}
        }
        try:
            with open(example_config_path, 'w') as f:
                yaml.dump(dummy_config, f, indent=2)
            print(f"Dummy config created at {example_config_path.resolve()}")
        except Exception as e:
            print(f"Error creating dummy config: {e}")

    print("Config structure defined. Loading/testing moved to main.py or caller.")