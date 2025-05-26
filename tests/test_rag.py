import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock, call

from knowledge_mcp.rag import (
    RagManager,
    RAGManagerError,
    UnsupportedProviderError,
    RAGInitializationError,
    ConfigurationError,
)
from knowledge_mcp.config import Config as AppConfig # Renamed to avoid conflict with pytest 'config'
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError, DEFAULT_QUERY_PARAMS, load_kb_query_config
from lightrag import LightRAG
from lightrag.base import QueryParam

# --- Fixtures ---

@pytest.fixture
def mock_app_config():
    """Provides a MagicMock for the application's Config object."""
    config = MagicMock(spec=AppConfig)

    # Mock the nested structure for lightrag settings
    config.lightrag = MagicMock()
    config.lightrag.llm = MagicMock()
    config.lightrag.llm.provider = "openai"
    config.lightrag.llm.model_name = "gpt-3.5-turbo"
    config.lightrag.llm.api_key = "test_llm_api_key"
    config.lightrag.llm.api_base = None
    config.lightrag.llm.max_token_size = 4096
    config.lightrag.llm.kwargs = {"temperature": 0.5}

    config.lightrag.embedding = MagicMock()
    config.lightrag.embedding.provider = "openai"
    config.lightrag.embedding.model_name = "text-embedding-ada-002"
    config.lightrag.embedding.api_key = "test_embedding_api_key" # Not directly used by funcs but good to have
    config.lightrag.embedding.api_base = None
    # embedding_dim, max_token_size are not directly used by RagManager for LightRAG init

    config.lightrag.embedding_cache = MagicMock()
    config.lightrag.embedding_cache.enabled = True
    config.lightrag.embedding_cache.similarity_threshold = 0.95
    
    return config

@pytest.fixture
def mock_kb_manager():
    """Provides a MagicMock for KnowledgeBaseManager."""
    manager = MagicMock(spec=KnowledgeBaseManager)
    # Default behavior for kb_exists, can be overridden in tests
    manager.kb_exists.return_value = True 
    manager.get_kb_path.return_value = Path("/tmp/fake_kb_path/kb_name")
    return manager

@pytest.fixture
def rag_manager(mock_app_config, mock_kb_manager):
    """Provides an initialized RagManager instance with mock dependencies."""
    # Even though RagManager takes AppConfig in __init__, it uses AppConfig.get_instance() internally.
    # So, we'll patch get_instance() for most tests.
    # The __init__ type hint is AppConfig, so we pass a mock AppConfig for instantiation.
    return RagManager(mock_app_config, mock_kb_manager)

@pytest.fixture
def mock_lightrag_instance():
    """Provides a mock LightRAG instance with async and sync mocked methods."""
    instance = MagicMock(spec=LightRAG)
    instance.initialize_storages = AsyncMock() # async method
    # query and ingest_doc are called via to_thread, so they are sync mocks
    instance.query = MagicMock(return_value="Mocked query result")
    instance.ingest_doc = MagicMock() 
    return instance

# --- Patches for external modules used by RagManager ---

@pytest.fixture(autouse=True)
def patch_openai_funcs():
    """Patches the OpenAI function imports to avoid actual API calls."""
    with patch('knowledge_mcp.openai_func.embedding_func', new_callable=MagicMock) as mock_embed_func, \
         patch('knowledge_mcp.openai_func.llm_model_func', new_callable=MagicMock) as mock_llm_func:
        yield {"embed_func": mock_embed_func, "llm_func": mock_llm_func}

@pytest.fixture(autouse=True)
def patch_initialize_pipeline_status():
    """Patches lightrag.kg.shared_storage.initialize_pipeline_status."""
    with patch('knowledge_mcp.rag.initialize_pipeline_status', new_callable=AsyncMock) as mock_init_status:
        yield mock_init_status

# --- Tests for RagManager ---

# 1. Initialization (__init__)
def test_ragmanager_init(mock_app_config, mock_kb_manager):
    """Test RagManager initializes correctly."""
    manager = RagManager(mock_app_config, mock_kb_manager)
    assert manager.config is mock_app_config
    assert manager.kb_manager is mock_kb_manager
    assert manager._rag_instances == {}

# 2. get_rag_instance (async)
@pytest.mark.asyncio
@patch('knowledge_mcp.rag.LightRAG') # Patch the class itself
@patch('knowledge_mcp.rag.Config.get_instance') # Patch the static method
async def test_get_rag_instance_successful_creation_and_caching(
    mock_config_get_instance,
    MockLightRAGClass,
    rag_manager,
    mock_kb_manager,
    mock_app_config, # The fixture for our mock config object
    mock_lightrag_instance, # The fixture for a pre-configured LightRAG instance mock
    patch_initialize_pipeline_status # Ensure this fixture is active
):
    """Test successful creation, initialization, and caching of LightRAG instance."""
    kb_name = "test_kb_create"
    mock_config_get_instance.return_value = mock_app_config # Config.get_instance() returns our mock
    MockLightRAGClass.return_value = mock_lightrag_instance # LightRAG() returns our mock instance

    # First call - should create and cache
    instance1 = await rag_manager.get_rag_instance(kb_name)

    mock_kb_manager.kb_exists.assert_called_with(kb_name) # Called by get_rag_instance and create_rag_instance
    mock_kb_manager.get_kb_path.assert_called_with(kb_name)
    MockLightRAGClass.assert_called_once_with(
        working_dir=str(mock_kb_manager.get_kb_path.return_value),
        llm_model_func=unittest.mock.ANY, # We patched openai_func, so it's a mock
        llm_model_kwargs=mock_app_config.lightrag.llm.kwargs,
        llm_model_name=mock_app_config.lightrag.llm.model_name,
        llm_model_max_token_size=mock_app_config.lightrag.llm.max_token_size,
        embedding_func=unittest.mock.ANY, # Also a mock from patch_openai_funcs
        embedding_cache_config={
            "enabled": mock_app_config.lightrag.embedding_cache.enabled,
            "similarity_threshold": mock_app_config.lightrag.embedding_cache.similarity_threshold,
        }
    )
    mock_lightrag_instance.initialize_storages.assert_awaited_once()
    patch_initialize_pipeline_status.assert_awaited_once() # Check this global init
    assert instance1 is mock_lightrag_instance
    assert kb_name in rag_manager._rag_instances
    assert rag_manager._rag_instances[kb_name] is mock_lightrag_instance

    # Second call - should return from cache
    # Reset mocks that should not be called again for a cached instance
    MockLightRAGClass.reset_mock()
    mock_lightrag_instance.initialize_storages.reset_mock()
    patch_initialize_pipeline_status.reset_mock()
    
    instance2 = await rag_manager.get_rag_instance(kb_name)
    assert instance2 is mock_lightrag_instance
    MockLightRAGClass.assert_not_called() # Should not create new LightRAG
    mock_lightrag_instance.initialize_storages.assert_not_awaited() # Should not re-initialize
    patch_initialize_pipeline_status.assert_not_awaited() # Should not re-initialize global status

@pytest.mark.asyncio
async def test_get_rag_instance_kb_not_found(rag_manager, mock_kb_manager):
    """Test KnowledgeBaseNotFoundError if kb_exists returns False."""
    kb_name = "ghost_kb"
    mock_kb_manager.kb_exists.return_value = False
    with pytest.raises(KnowledgeBaseNotFoundError, match=f"Knowledge base '{kb_name}' does not exist."):
        await rag_manager.get_rag_instance(kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_config_error_missing_llm(mock_config_get_instance, rag_manager, mock_app_config):
    """Test ConfigurationError if config.lightrag.llm is missing."""
    kb_name = "config_fail_kb"
    # Simulate missing llm config section
    type(mock_app_config.lightrag).llm = PropertyMock(side_effect=AttributeError) # More robust way to simulate missing
    # Or simpler: del mock_app_config.lightrag.llm # if the mock allows deletion
    mock_config_get_instance.return_value = mock_app_config

    with pytest.raises(ConfigurationError, match="Language model settings .* are missing."):
        await rag_manager.get_rag_instance(kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_config_error_missing_embedding(mock_config_get_instance, rag_manager, mock_app_config):
    """Test ConfigurationError if config.lightrag.embedding is missing."""
    kb_name = "config_fail_kb_embed"
    type(mock_app_config.lightrag).embedding = PropertyMock(side_effect=AttributeError)
    mock_config_get_instance.return_value = mock_app_config
    with pytest.raises(ConfigurationError, match="Embedding model settings .* are missing."):
        await rag_manager.get_rag_instance(kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_unsupported_llm_provider(mock_config_get_instance, rag_manager, mock_app_config):
    """Test UnsupportedProviderError for LLM."""
    kb_name = "unsupported_llm_kb"
    mock_app_config.lightrag.llm.provider = "bard" # Unsupported
    mock_config_get_instance.return_value = mock_app_config
    with pytest.raises(UnsupportedProviderError, match="Only OpenAI language model provider currently supported."):
        await rag_manager.get_rag_instance(kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_unsupported_embedding_provider(mock_config_get_instance, rag_manager, mock_app_config):
    """Test UnsupportedProviderError for embedding."""
    kb_name = "unsupported_embed_kb"
    mock_app_config.lightrag.embedding.provider = "vertexai" # Unsupported
    mock_config_get_instance.return_value = mock_app_config
    with pytest.raises(UnsupportedProviderError, match="Only OpenAI embedding provider currently supported."):
        await rag_manager.get_rag_instance(kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.LightRAG')
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_lightrag_init_fails(mock_config_get_instance, MockLightRAGClass, rag_manager, mock_app_config):
    """Test RAGInitializationError if LightRAG() instantiation fails."""
    kb_name = "lightrag_init_fail_kb"
    mock_config_get_instance.return_value = mock_app_config
    MockLightRAGClass.side_effect = Exception("LightRAG constructor failed")
    with pytest.raises(RAGInitializationError, match=f"Failed to initialize LightRAG for {kb_name}: LightRAG constructor failed"):
        await rag_manager.get_rag_instance(kb_name)


@pytest.mark.asyncio
@patch('knowledge_mcp.rag.LightRAG')
@patch('knowledge_mcp.rag.Config.get_instance')
async def test_get_rag_instance_initialize_storages_fails(mock_config_get_instance, MockLightRAGClass, rag_manager, mock_app_config, mock_lightrag_instance):
    """Test RAGInitializationError if rag.initialize_storages() fails."""
    kb_name = "storage_init_fail_kb"
    mock_config_get_instance.return_value = mock_app_config
    MockLightRAGClass.return_value = mock_lightrag_instance
    mock_lightrag_instance.initialize_storages.side_effect = Exception("initialize_storages failed")
    
    with pytest.raises(RAGInitializationError, match=f"Failed to initialize LightRAG for {kb_name}: initialize_storages failed"):
        await rag_manager.get_rag_instance(kb_name)


# 3. create_rag_instance (mostly covered, but direct test for specific logic if any)
# The tests for get_rag_instance are quite comprehensive for create_rag_instance's logic.

# 4. remove_rag_instance
@pytest.mark.asyncio
async def test_remove_rag_instance_success(rag_manager, mock_lightrag_instance):
    """Test successful removal of a cached RAG instance."""
    kb_name = "cached_kb"
    # Manually add to cache for testing removal
    rag_manager._rag_instances[kb_name] = mock_lightrag_instance
    
    rag_manager.remove_rag_instance(kb_name)
    assert kb_name not in rag_manager._rag_instances

def test_remove_rag_instance_not_in_cache(rag_manager, caplog):
    """Test removing an instance not in cache (raises KnowledgeBaseNotFoundError)."""
    kb_name = "not_cached_kb"
    with pytest.raises(KnowledgeBaseNotFoundError, match=f"Knowledge base '{kb_name}' not found."):
        rag_manager.remove_rag_instance(kb_name)

def test_remove_rag_instance_no_name_provided(rag_manager):
    """Test ValueError if kb_name is not provided."""
    with pytest.raises(ValueError, match="Knowledgebase name is required."):
        rag_manager.remove_rag_instance(None)
    with pytest.raises(ValueError, match="Knowledgebase name is required."):
        rag_manager.remove_rag_instance()


# 5. query (async)
@pytest.mark.asyncio
@patch('knowledge_mcp.rag.load_kb_query_config')
async def test_query_success(
    mock_load_config,
    rag_manager,
    mock_kb_manager,
    mock_lightrag_instance
):
    """Test successful query execution with merged config."""
    kb_name = "query_kb"
    query_text = "What is the meaning of life?"
    kb_specific_config = {"mode": "vector", "top_k": 10} # No description
    runtime_kwargs = {"top_k": 5, "response_type": "list"} # top_k overrides kb_specific

    # Mock get_rag_instance to return our mock LightRAG instance
    # This bypasses the actual RAG creation logic for this test.
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    
    mock_load_config.return_value = kb_specific_config
    
    expected_query_result = "Mocked query result: 42"
    mock_lightrag_instance.query.return_value = expected_query_result # query is sync

    result = await rag_manager.query(kb_name, query_text, **runtime_kwargs)

    rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)
    mock_kb_manager.get_kb_path.assert_called_once_with(kb_name)
    mock_load_config.assert_called_once_with(mock_kb_manager.get_kb_path.return_value)
    
    # Check QueryParam construction
    # Expected final params: mode from kb_specific, top_k from runtime, response_type from runtime
    # Other defaults from DEFAULT_QUERY_PARAMS (excluding 'description')
    final_params_for_qp = DEFAULT_QUERY_PARAMS.copy()
    del final_params_for_qp["description"] # description should be removed
    final_params_for_qp.update(kb_specific_config) # Apply KB config
    final_params_for_qp.update(runtime_kwargs)   # Apply runtime overrides
    
    # mock_lightrag_instance.query is called via to_thread, check its args
    call_args = mock_lightrag_instance.query.call_args
    assert call_args is not None
    assert call_args[1]['query'] == query_text # query is a kwarg for the method passed to to_thread
    
    # Validate the QueryParam object passed to rag_instance.query
    param_instance_arg = call_args[1]['param']
    assert isinstance(param_instance_arg, QueryParam)
    for key, value in final_params_for_qp.items():
        assert getattr(param_instance_arg, key) == value
        
    assert result == expected_query_result


@pytest.mark.asyncio
@patch('knowledge_mcp.rag.load_kb_query_config')
async def test_query_load_kb_query_config_error(mock_load_config, rag_manager, mock_lightrag_instance):
    """Test query fails if load_kb_query_config raises an error (e.g. ConfigurationError)."""
    kb_name = "query_config_fail_kb"
    query_text = "test query"
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    mock_load_config.side_effect = ConfigurationError("Failed to load KB config")

    with pytest.raises(ConfigurationError, match="Failed to load KB config"):
        await rag_manager.query(kb_name, query_text)

@pytest.mark.asyncio
async def test_query_get_rag_instance_fails(rag_manager):
    """Test query propagates error from get_rag_instance."""
    kb_name = "rag_fail_kb"
    query_text = "test query"
    rag_manager.get_rag_instance = AsyncMock(side_effect=RAGInitializationError("RAG init failed for query"))

    with pytest.raises(RAGInitializationError, match="RAG init failed for query"):
        await rag_manager.query(kb_name, query_text)

@pytest.mark.asyncio
@patch('knowledge_mcp.rag.load_kb_query_config')
async def test_query_lightrag_query_fails(mock_load_config, rag_manager, mock_lightrag_instance):
    """Test RAGManagerError wrapping if LightRAG.query fails."""
    kb_name = "lightrag_query_fail_kb"
    query_text = "test query"
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    mock_load_config.return_value = {} # Minimal config
    mock_lightrag_instance.query.side_effect = Exception("Underlying LightRAG query error")

    with pytest.raises(RAGManagerError, match="Async query failed: Underlying LightRAG query error"):
        await rag_manager.query(kb_name, query_text)

# 6. ingest_document (async)
@pytest.mark.asyncio
async def test_ingest_document_success(rag_manager, mock_lightrag_instance, tmp_path):
    """Test successful document ingestion."""
    kb_name = "ingest_kb"
    doc_content = "This is a test document."
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(doc_content)
    
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    
    # Call ingest_document
    result_doc_id = await rag_manager.ingest_document(kb_name, doc_path, doc_id="custom_id")

    rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)
    # ingest_doc is sync, called via to_thread
    mock_lightrag_instance.ingest_doc.assert_called_once_with(doc_path=str(doc_path))
    assert result_doc_id == "custom_id" # Should return the provided doc_id

@pytest.mark.asyncio
async def test_ingest_document_default_doc_id(rag_manager, mock_lightrag_instance, tmp_path):
    """Test default doc_id generation (filename stem)."""
    kb_name = "ingest_kb_default_id"
    doc_path = tmp_path / "another_doc.md"
    doc_path.write_text("# Markdown Content")
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    result_doc_id = await rag_manager.ingest_document(kb_name, doc_path)
    
    mock_lightrag_instance.ingest_doc.assert_called_once_with(doc_path=str(doc_path))
    assert result_doc_id == "another_doc" # Stem of the filename

@pytest.mark.asyncio
async def test_ingest_document_get_rag_instance_fails(rag_manager, tmp_path):
    """Test ingest_document propagates error from get_rag_instance."""
    kb_name = "ingest_rag_fail_kb"
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("content")
    rag_manager.get_rag_instance = AsyncMock(side_effect=RAGInitializationError("RAG init failed for ingest"))

    with pytest.raises(RAGInitializationError, match="RAG init failed for ingest"):
        await rag_manager.ingest_document(kb_name, doc_path)

@pytest.mark.asyncio
async def test_ingest_document_file_not_found(rag_manager, mock_lightrag_instance, tmp_path):
    """Test FileNotFoundError if document does not exist (if RagManager checks).
       Currently, ingest_document itself doesn't check for file existence before calling
       rag_instance.ingest_doc. This test assumes LightRAG's ingest_doc might raise it,
       or if RagManager were to add a check. For now, mocking ingest_doc to raise it.
    """
    kb_name = "ingest_file_fail_kb"
    non_existent_doc_path = tmp_path / "ghost_doc.txt"
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    # Simulate ingest_doc failing because file not found
    mock_lightrag_instance.ingest_doc.side_effect = FileNotFoundError(f"File not found: {non_existent_doc_path}")

    with pytest.raises(FileNotFoundError): # Test if it propagates directly
        await rag_manager.ingest_document(kb_name, non_existent_doc_path)

@pytest.mark.asyncio
async def test_ingest_document_lightrag_ingest_fails(rag_manager, mock_lightrag_instance, tmp_path):
    """Test RAGManagerError wrapping if LightRAG.ingest_doc fails."""
    kb_name = "lightrag_ingest_fail_kb"
    doc_path = tmp_path / "real_doc.txt"
    doc_path.write_text("content")
    rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    mock_lightrag_instance.ingest_doc.side_effect = Exception("Underlying LightRAG ingest error")

    with pytest.raises(RAGManagerError, match=f"Ingestion failed for '{doc_path.name}': Underlying LightRAG ingest error"):
        await rag_manager.ingest_document(kb_name, doc_path)

"""
This test suite covers the RagManager class from knowledge_mcp.rag.

Key areas tested:
- Initialization: Correct setup with Config and KnowledgeBaseManager.
- get_rag_instance (async):
    - Successful creation, configuration (OpenAI provider), initialization (LightRAG, pipeline_status), and caching.
    - Error handling: KnowledgeBaseNotFoundError, ConfigurationError (missing sections), UnsupportedProviderError, RAGInitializationError (LightRAG instantiation or initialize_storages failure).
- remove_rag_instance:
    - Successful removal from cache.
    - KnowledgeBaseNotFoundError for non-cached KB.
    - ValueError for missing kb_name.
- query (async):
    - Successful query execution with merged KB-specific and runtime configurations.
    - Correct QueryParam construction.
    - Error handling: Propagation of errors from get_rag_instance, ConfigurationError from load_kb_query_config/QueryParam, and RAGManagerError wrapping LightRAG.query failures.
- ingest_document (async):
    - Successful document ingestion.
    - Correct call to LightRAG.ingest_doc with doc_path.
    - Default and custom doc_id handling.
    - Error handling: Propagation from get_rag_instance, FileNotFoundError (simulated from LightRAG.ingest_doc), and RAGManagerError wrapping LightRAG.ingest_doc failures.

Utilized pytest.mark.asyncio, tmp_path, extensive unittest.mock (MagicMock, AsyncMock, patch, PropertyMock),
and fixtures for dependencies. External modules like openai_func and LightRAG components are mocked.
"""
