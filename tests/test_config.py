import yaml
import pytest
from pathlib import Path
from pydantic import ValidationError
from knowledge_mcp.config import (
    ConfigService, 
    Config, 
    KnowledgeBaseConfig,
    EmbeddingModelConfig,
    LanguageModelConfig,
    LoggingConfig,
)

@pytest.fixture(autouse=True)
def reset_config_service_singleton():
    """Ensures each test gets a fresh ConfigService state."""
    ConfigService._instance = None
    ConfigService._initialized = False
    ConfigService._config_data = None
    ConfigService._config_path = None
    yield 
    ConfigService._instance = None
    ConfigService._initialized = False
    ConfigService._config_data = None
    ConfigService._config_path = None

def test_load_config_success(tmp_path: Path, monkeypatch):
    """Tests successful loading and env var substitution via ConfigService."""
    config_data = {
        'knowledge_base': {'base_dir': '/tmp/kb_test'},
        'embedding_model': {
            'provider': 'test_emb_provider',
            'base_url': 'http://test.emb.com',
            'model_name': 'test-embedding-model',
            'api_key': '${TEST_API_KEY}'
        },
        'language_model': {
            'provider': 'test_llm_provider',
            'base_url': 'http://test.llm.com',
            'model_name': 'test-llm-model',
            'api_key': '${TEST_API_KEY}'
        },
        'logging': {'level': 'DEBUG', 'file': str(tmp_path / 'test.log')}
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    monkeypatch.setenv('TEST_API_KEY', 'secret_test_value')

    config_service = ConfigService.get_instance(str(config_file))

    assert config_service.config_file_path == config_file.resolve()
    assert config_service.knowledge_base == KnowledgeBaseConfig(base_dir=Path('/tmp/kb_test'))
    assert config_service.embedding_model == EmbeddingModelConfig(
        provider='test_emb_provider',
        base_url='http://test.emb.com',
        model_name='test-embedding-model',
        api_key='secret_test_value' 
    )
    assert config_service.language_model == LanguageModelConfig(
        provider='test_llm_provider',
        base_url='http://test.llm.com',
        model_name='test-llm-model',
        api_key='secret_test_value' 
    )
    assert config_service.logging == LoggingConfig(level='DEBUG', file=tmp_path / 'test.log')

    config_service_2 = ConfigService.get_instance("some_other_path.yaml") 
    assert config_service is config_service_2
    assert config_service_2.config_file_path == config_file.resolve() 

def test_missing_section(tmp_path: Path):
    """Tests validation failure for missing required config section."""
    config_data = {
        'knowledge_base': {'base_dir': '/tmp/kb'},
        'embedding_model': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'api_key': 'dummy_key'
        },
        'logging': {'level': 'INFO'}
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(RuntimeError) as excinfo:
        ConfigService.get_instance(str(config_file))
    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert 'language_model' in str(excinfo.value.__cause__)
    assert 'Field required' in str(excinfo.value.__cause__)

def test_invalid_data_type(tmp_path: Path):
    """Tests validation failure for incorrect data type."""
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
        'logging': {'level': 123, 'file': '/tmp/log.log'} 
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_data))

    with pytest.raises(RuntimeError) as excinfo:
        ConfigService.get_instance(str(config_file))
    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert 'logging.level' in str(excinfo.value.__cause__)
    assert 'Input should be a valid string' in str(excinfo.value.__cause__)


def test_file_not_found():
    """Tests that FileNotFoundError during init raises RuntimeError."""
    with pytest.raises(RuntimeError) as excinfo:
        ConfigService.get_instance('nonexistent_config.yaml')
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert 'nonexistent_config.yaml' in str(excinfo.value.__cause__)

def test_invalid_yaml(tmp_path: Path):
    """Tests that YAMLError during init raises RuntimeError."""
    config_file = tmp_path / 'invalid_config.yaml'
    config_file.write_text("knowledge_base: { base_dir: /tmp/kb\nembedding_model: [invalid yaml")

    with pytest.raises(RuntimeError) as excinfo:
        ConfigService.get_instance(str(config_file))
    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)
