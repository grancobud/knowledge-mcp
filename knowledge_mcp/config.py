"""
Configuration parsing with YAML support and environment variable substitution.
"""
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Type, TypeVar


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


def _load_and_validate_config(config_path: str) -> Config:
    """
    Internal function to load YAML, substitute env vars, and validate with Pydantic.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found or is not a file: {config_path}")

    print(f"Reading config from: {config_file.resolve()}")
    raw = config_file.read_text()
    
    substituted_raw = raw
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        if placeholder in substituted_raw:
            print(f"Substituting {placeholder}...")
            substituted_raw = substituted_raw.replace(placeholder, value)

    try:
        config_dict = yaml.safe_load(substituted_raw)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    if not isinstance(config_dict, dict):
        print(f"Invalid YAML format in {config_path}: Expected a dictionary, got {type(config_dict)}")
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: Expected a dictionary.")

    try:
        validated_config = Config.model_validate(config_dict)
        print("Config validation successful.")
    except ValidationError as e:
        print(f"Configuration validation failed for {config_path}:\n{e}")
        raise e

    return validated_config


TConfigService = TypeVar('TConfigService', bound='ConfigService')


class ConfigService:
    """Singleton service to load and provide access to the validated Config object."""
    _instance: Optional[TConfigService] = None
    _config_data: Optional[Config] = None
    _config_path: Optional[Path] = None
    _initialized: bool = False

    def __new__(cls: Type[TConfigService], config_path: str = "config.yaml") -> TConfigService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config_path)
        elif not cls._initialized:
             print("ConfigService: Re-initializing potentially failed instance reference...")
             cls._instance._initialize(config_path)
        elif Path(config_path).resolve() != cls._config_path:
             print(f"ConfigService INFO: Instance already initialized with {cls._config_path}. "
                   f"Ignoring new path {config_path}.")

        return cls._instance

    def _initialize(self, config_path: str):
        """Internal method to load config data, called only once per instance lifetime."""
        if self._initialized:
            return

        print(f"ConfigService: Initializing with config path: {config_path}")
        try:
            absolute_path = Path(config_path).resolve()
            self._config_data = _load_and_validate_config(str(absolute_path))
            self._config_path = absolute_path
            self._initialized = True
            print(f"ConfigService: Configuration loaded successfully from {self._config_path}")
        except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
            print(f"ConfigService: FATAL - Failed to load or validate configuration from '{config_path}': {e}")
            self._config_data = None
            self._initialized = False 
            self._config_path = None
            raise RuntimeError(f"Failed to initialize ConfigService: {e}") from e
        except Exception as e:
            print(f"ConfigService: FATAL - An unexpected error occurred during initialization: {e}")
            self._config_data = None
            self._initialized = False
            self._config_path = None
            raise RuntimeError(f"Unexpected error during ConfigService initialization: {e}") from e


    @classmethod
    def get_instance(cls: Type[TConfigService], config_path: str = "config.yaml") -> TConfigService:
        """Gets the singleton instance of the ConfigService.

        Args:
            config_path: Path to the config file. Used ONLY on the first call
                         to create the instance. Subsequent calls ignore this.

        Returns:
            The singleton ConfigService instance.
        
        Raises:
            RuntimeError: If configuration loading fails during the first initialization.
        """
        return cls(config_path)

    def get_config_data(self) -> Config:
        """Returns the loaded and validated Pydantic Config object."""
        if not self._initialized or self._config_data is None:
            raise RuntimeError("ConfigService is not initialized. Configuration may have failed to load.")
        return self._config_data

    @property
    def knowledge_base(self) -> KnowledgeBaseConfig:
        return self.get_config_data().knowledge_base

    @property
    def embedding_model(self) -> EmbeddingModelConfig:
        return self.get_config_data().embedding_model

    @property
    def language_model(self) -> LanguageModelConfig:
        return self.get_config_data().language_model

    @property
    def logging(self) -> LoggingConfig:
        return self.get_config_data().logging

    @property
    def config_file_path(self) -> Optional[Path]:
        """Returns the absolute path of the configuration file successfully used."""
        return self._config_path


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

    try:
        config_service1 = ConfigService.get_instance()
        print(f"Instance 1 Config Path: {config_service1.config_file_path}")

        loaded_config = config_service1.get_config_data()
        print(f"KB Base Dir (method): {loaded_config.knowledge_base.base_dir}")
        print(f"LLM Provider (method): {loaded_config.language_model.provider}")

        print(f"KB Base Dir (prop): {config_service1.knowledge_base.base_dir}")
        print(f"LLM Provider (prop): {config_service1.language_model.provider}")
        print(f"Logging Level (prop): {config_service1.logging.level}")

        config_service2 = ConfigService.get_instance("non_existent_config.yaml") 
        print(f"Instance 1 == Instance 2: {config_service1 is config_service2}")
        print(f"Instance 2 Config Path (still): {config_service2.config_file_path}")

        api_key = config_service1.embedding_model.api_key
        print(f"Embedding API Key: {api_key}") 

    except RuntimeError as e:
        print(f"\n--- RUNTIME ERROR DURING EXAMPLE USAGE --- ")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR DURING EXAMPLE USAGE --- ")
        print(f"Error: {e}")