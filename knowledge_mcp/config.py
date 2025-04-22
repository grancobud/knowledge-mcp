"""
Configuration parsing with YAML support and environment variable substitution.
"""
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


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


class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="Logging level")
    file: Optional[Path] = Field(None, description="Path to the log file")


class Config(BaseModel):
    knowledge_base: KnowledgeBaseConfig
    embedding_model: EmbeddingModelConfig
    language_model: LanguageModelConfig
    logging: LoggingConfig


def load_config(config_path: str) -> Config:
    """
    Load YAML configuration file, substitute environment variables, and parse into Config model.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed and validated configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
        ValidationError: If the configuration data is invalid.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = config_file.read_text()
    # Substitute environment variables of the form ${VAR_NAME}
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    try:
        config_dict = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    if not isinstance(config_dict, dict):
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: Expected a dictionary.")

    try:
        config = Config.model_validate(config_dict)
    except ValidationError as e:
        # Re-raise the original validation error to preserve details
        raise e

    return config
