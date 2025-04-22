import yaml
import pytest
from pathlib import Path
from pydantic import ValidationError
from knowledge_mcp.config import load_config, Config, KnowledgeBaseConfig, EmbeddingModelConfig, LanguageModelConfig, LoggingConfig


def test_load_config(tmp_path: Path, monkeypatch):
    # Create a sample YAML config with env var placeholder
    config_data = {
        'knowledge_base': {'base_dir': '/tmp/kb'},
        'embedding_model': {
            'provider': 'openai',
            'base_url': 'https://api.openai.com/v1',
            'model_name': 'text-embedding-3-small',
            'api_key': '${TEST_API_KEY}'
        },
        'language_model': {
            'provider': 'openai',
            'base_url': 'https://api.openai.com/v1',
            'model_name': 'gpt-4o',
            'api_key': '${TEST_API_KEY}'
        },
        'logging': {'level': 'INFO', 'file': '/tmp/log.log'}
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    # Set environment variable
    monkeypatch.setenv('TEST_API_KEY', 'secret_value')

    # Load config
    loaded_config = load_config(str(config_file))

    # Assertions using the Pydantic model
    assert isinstance(loaded_config, Config)
    assert loaded_config.knowledge_base == KnowledgeBaseConfig(base_dir=Path('/tmp/kb'))
    assert loaded_config.embedding_model == EmbeddingModelConfig(
        provider='openai',
        base_url='https://api.openai.com/v1',
        model_name='text-embedding-3-small',
        api_key='secret_value'
    )
    assert loaded_config.language_model == LanguageModelConfig(
        provider='openai',
        base_url='https://api.openai.com/v1',
        model_name='gpt-4o',
        api_key='secret_value'
    )
    assert loaded_config.logging == LoggingConfig(level='INFO', file=Path('/tmp/log.log'))


def test_missing_section(tmp_path: Path):
    # Create config missing a required section (e.g., language_model)
    config_data = {
        'knowledge_base': {'base_dir': '/tmp/kb'},
        'embedding_model': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'api_key': 'dummy_key'
        },
        # 'language_model': { ... } is missing
        'logging': {'level': 'INFO'}
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(ValidationError) as exc:
        load_config(str(config_file))
    assert 'language_model' in str(exc.value)
    assert 'Field required' in str(exc.value)

def test_invalid_data_type(tmp_path: Path):
    # Create config with invalid data type (e.g., logging level as int)
    config_data = {
        'knowledge_base': {'base_dir': '/tmp/kb'},
        'embedding_model': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'api_key': 'dummy_key'
        },
        'language_model': {
            'provider': 'openai',
            'model_name': 'gpt-4o',
            'api_key': 'dummy_key'
        },
        'logging': {'level': 123, 'file': '/tmp/log.log'} # level should be string
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(ValidationError) as exc:
        load_config(str(config_file))
    assert 'logging.level' in str(exc.value)
    assert 'Input should be a valid string' in str(exc.value)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent.yaml')

def test_invalid_yaml(tmp_path: Path):
    config_file = tmp_path / 'invalid_config.yaml'
    config_file.write_text("knowledge_base: { base_dir: /tmp/kb\nembedding_model: [invalid yaml")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))
