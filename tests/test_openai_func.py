import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

# Module to be tested (will be imported after patching Config)
# import knowledge_mcp.openai_func as openai_func_module

from knowledge_mcp.config import Config
from lightrag.utils import EmbeddingFunc # For type checking

# --- Fixtures ---

@pytest.fixture
def mock_config_instance():
    """Provides a MagicMock for a Config instance with OpenAI settings."""
    config = MagicMock(spec=Config)
    
    config.lightrag = MagicMock()
    
    config.lightrag.llm = MagicMock()
    config.lightrag.llm.provider = "openai" # Assumed by the module
    config.lightrag.llm.model_name = "gpt-test-from-config" # This is NOT used by llm_model_func
    config.lightrag.llm.api_key = "test_llm_api_key_from_config"
    config.lightrag.llm.api_base = "https://config.openai.azure.com/"
    
    config.lightrag.embedding = MagicMock()
    config.lightrag.embedding.provider = "openai" # Assumed
    config.lightrag.embedding.model_name = "text-embedding-test-from-config"
    config.lightrag.embedding.api_key = "test_embedding_api_key_from_config"
    config.lightrag.embedding.api_base = "https://config.embedding.azure.com/"
    config.lightrag.embedding.embedding_dim = 1536
    config.lightrag.embedding.max_token_size = 8192
    
    return config

@pytest.fixture(autouse=True)
def patch_config_get_instance(mock_config_instance):
    """Patches Config.get_instance() for all tests in this module *before* openai_func is imported by tests."""
    with patch('knowledge_mcp.config.Config.get_instance', return_value=mock_config_instance) as mock_get:
        # Reload the openai_func module to ensure it picks up the patched Config
        # This is crucial for testing the import-time initialization of `embedding_func`
        global openai_func_module
        import importlib
        if 'knowledge_mcp.openai_func' in sys.modules:
            openai_func_module = importlib.reload(sys.modules['knowledge_mcp.openai_func'])
        else:
            import knowledge_mcp.openai_func as openai_func_module
        yield mock_get


@pytest.fixture
def mock_openai_complete():
    """Patches lightrag.llm.openai.openai_complete_if_cache."""
    with patch('knowledge_mcp.openai_func.openai_complete_if_cache', new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = "Mocked LLM response"
        yield mock_complete

@pytest.fixture
def mock_openai_embed():
    """Patches lightrag.llm.openai.openai_embed."""
    with patch('knowledge_mcp.openai_func.openai_embed', new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
        yield mock_embed

# Need to import sys for the reload logic in patch_config_get_instance
import sys

# --- Tests for llm_model_func ---

@pytest.mark.asyncio
async def test_llm_model_func_success(mock_openai_complete, mock_config_instance):
    """Test llm_model_func successfully calls openai_complete_if_cache with correct params."""
    prompt_text = "Hello, world!"
    system_prompt_text = "You are a test AI."
    
    # Mock the global_config structure within kwargs
    mock_hashing_kv = MagicMock()
    mock_hashing_kv.global_config = {"llm_model_name": "gpt-test-from-kwargs"}
    
    kwargs_to_pass = {
        "hashing_kv": mock_hashing_kv,
        "temperature": 0.7,
        # api_key and base_url should be picked from mock_config_instance
    }

    response = await openai_func_module.llm_model_func(
        prompt_text,
        system_prompt=system_prompt_text,
        **kwargs_to_pass
    )

    assert response == "Mocked LLM response"
    mock_openai_complete.assert_awaited_once_with(
        model="gpt-test-from-kwargs", # From kwargs.hashing_kv.global_config
        prompt=prompt_text,
        system_prompt=system_prompt_text,
        history_messages=[], # Default
        api_key=mock_config_instance.lightrag.llm.api_key, # From Config
        base_url=mock_config_instance.lightrag.llm.api_base, # From Config
        keyword_extraction=False, # Default
        temperature=0.7 # From kwargs_to_pass
    )

@pytest.mark.asyncio
async def test_llm_model_func_missing_api_key(mock_config_instance):
    """Test llm_model_func raises ValueError if API key is missing."""
    mock_config_instance.lightrag.llm.api_key = None # Simulate missing API key
    
    mock_hashing_kv = MagicMock()
    mock_hashing_kv.global_config = {"llm_model_name": "any_model"}

    with pytest.raises(ValueError, match="OpenAI API key is not provided in config"):
        await openai_func_module.llm_model_func("A prompt", hashing_kv=mock_hashing_kv)

@pytest.mark.asyncio
async def test_llm_model_func_no_api_base(mock_openai_complete, mock_config_instance):
    """Test llm_model_func works correctly when api_base is None in config."""
    mock_config_instance.lightrag.llm.api_base = None
    
    mock_hashing_kv = MagicMock()
    mock_hashing_kv.global_config = {"llm_model_name": "a_model"}

    await openai_func_module.llm_model_func("A test prompt", hashing_kv=mock_hashing_kv)
    
    mock_openai_complete.assert_awaited_once()
    call_kwargs = mock_openai_complete.call_args.kwargs
    assert call_kwargs['base_url'] is None


# --- Tests for openai_embedding_func ---

@pytest.mark.asyncio
async def test_openai_embedding_func_success(mock_openai_embed, mock_config_instance):
    """Test openai_embedding_func successfully calls openai_embed with correct params."""
    texts_to_embed = ["text1", "text2"]
    
    result_array = await openai_func_module.openai_embedding_func(texts_to_embed)

    assert isinstance(result_array, np.ndarray)
    mock_openai_embed.assert_awaited_once_with(
        texts=texts_to_embed,
        model=mock_config_instance.lightrag.embedding.model_name, # From Config
        api_key=mock_config_instance.lightrag.embedding.api_key, # From Config
        base_url=mock_config_instance.lightrag.embedding.api_base # From Config
    )

@pytest.mark.asyncio
async def test_openai_embedding_func_no_api_base(mock_openai_embed, mock_config_instance):
    """Test openai_embedding_func works correctly when api_base is None in config for embeddings."""
    mock_config_instance.lightrag.embedding.api_base = None
    texts_to_embed = ["text3"]

    await openai_func_module.openai_embedding_func(texts_to_embed)

    mock_openai_embed.assert_awaited_once()
    call_kwargs = mock_openai_embed.call_args.kwargs
    assert call_kwargs['base_url'] is None
    assert call_kwargs['model'] == mock_config_instance.lightrag.embedding.model_name


# --- Tests for embedding_func (EmbeddingFunc wrapper) ---

def test_embedding_func_instance_properties(mock_config_instance):
    """Test that the global embedding_func instance has correct properties set at import time."""
    # openai_func_module is reloaded by patch_config_get_instance fixture
    # so embedding_func should have been created with the mocked config values
    
    assert isinstance(openai_func_module.embedding_func, EmbeddingFunc)
    assert openai_func_module.embedding_func.embedding_dim == mock_config_instance.lightrag.embedding.embedding_dim
    assert openai_func_module.embedding_func.max_token_size == mock_config_instance.lightrag.embedding.max_token_size
    assert openai_func_module.embedding_func.func == openai_func_module.openai_embedding_func

@pytest.mark.asyncio
async def test_embedding_func_call_forwards_to_openai_embedding_func(mock_openai_embed, mock_config_instance):
    """Test calling the embedding_func instance correctly calls the wrapped openai_embedding_func."""
    texts = ["call_test"]
    
    # The embedding_func itself is not async, but its wrapped func is.
    # However, EmbeddingFunc.__call__ makes it awaitable.
    await openai_func_module.embedding_func(texts) 

    # Check if openai_embedding_func (which is mocked by mock_openai_embed indirectly via func=openai_embedding_func)
    # was called with the right parameters from config.
    mock_openai_embed.assert_awaited_once_with(
        texts=texts,
        model=mock_config_instance.lightrag.embedding.model_name,
        api_key=mock_config_instance.lightrag.embedding.api_key,
        base_url=mock_config_instance.lightrag.embedding.api_base
    )

"""
This test suite covers `knowledge_mcp/openai_func.py`.

Key areas tested:
1.  **`llm_model_func`**:
    *   Verifies successful calls to `lightrag.llm.openai.openai_complete_if_cache`.
    *   Ensures `model_name` is correctly retrieved from `kwargs["hashing_kv"].global_config`.
    *   Ensures `api_key` and `api_base` are correctly retrieved from `Config.get_instance()`.
    *   Tests `ValueError` is raised if `api_key` is missing from config.
    *   Checks correct behavior when `api_base` is `None`.

2.  **`openai_embedding_func`**:
    *   Verifies successful calls to `lightrag.llm.openai.openai_embed`.
    *   Ensures `model_name`, `api_key`, and `api_base` for embeddings are correctly retrieved from `Config.get_instance()`.
    *   Checks correct behavior when `api_base` for embeddings is `None`.

3.  **`embedding_func` (EmbeddingFunc wrapper instance)**:
    *   Tests that the global `embedding_func` instance (created at import time) has its `embedding_dim` and `max_token_size` properties correctly set from the (mocked) `Config` during module import/reload.
    *   Verifies that calling `embedding_func` correctly invokes the wrapped `openai_embedding_func`, which in turn calls the mocked `lightrag.llm.openai.openai_embed` with parameters derived from `Config`.

The `patch_config_get_instance` autouse fixture is crucial: it patches `Config.get_instance()`
and reloads the `knowledge_mcp.openai_func` module. This ensures that the `embedding_func`
instance, which is created at module import time, picks up the mocked configuration values
for its `embedding_dim` and `max_token_size` properties.
"""
