import pytest
import json
from textwrap import dedent
from unittest.mock import patch, MagicMock, AsyncMock, call

from knowledge_mcp.mcp_server import MCP
from knowledge_mcp.rag import RagManager, RAGManagerError, ConfigurationError
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError, KnowledgeBaseError
from fastmcp import FastMCP # Used for patching

# --- Fixtures ---

@pytest.fixture
def mock_rag_manager():
    """Provides a MagicMock for RagManager."""
    manager = MagicMock(spec=RagManager)
    manager.query = AsyncMock() # query is an async method
    return manager

@pytest.fixture
def mock_kb_manager():
    """Provides a MagicMock for KnowledgeBaseManager."""
    manager = MagicMock(spec=KnowledgeBaseManager)
    manager.list_kbs = AsyncMock() # list_kbs is an async method
    return manager

@pytest.fixture
@patch('knowledge_mcp.mcp_server.FastMCP.run') # Patch run for all MCP instances in tests
def mcp_instance(mock_fastmcp_run_method, mock_rag_manager, mock_kb_manager):
    """
    Provides an initialized MCP instance with FastMCP.run patched.
    The patch is passed automatically by pytest if named the same as the mock object.
    """
    return MCP(rag_manager=mock_rag_manager, kb_manager=mock_kb_manager)

# --- Tests for MCP ---

# 1. Initialization (__init__)
@patch('knowledge_mcp.mcp_server.FastMCP.run') # Patch FastMCP.run specifically for this test
@patch('knowledge_mcp.mcp_server.FastMCP.__init__') # Patch FastMCP.__init__ to inspect its args
def test_mcp_init_success(
    mock_fastmcp_init,
    mock_fastmcp_run,
    mock_rag_manager,
    mock_kb_manager
):
    """Test successful MCP initialization, FastMCP instantiation, tool addition, and run call."""
    # Store the actual FastMCP instance created inside MCP to check add_tool calls
    real_fastmcp_instance_storage = {} 
    def capture_instance(self_fmc, name, instructions):
        real_fastmcp_instance_storage['instance'] = self_fmc # Store the actual FastMCP instance
        # Call original __init__ if needed, or just set attributes
        self_fmc.name = name
        self_fmc.instructions = instructions
        self_fmc.tools = {} # Simulate tools dict
        self_fmc.add_tool = MagicMock() # Mock add_tool on the captured instance

    mock_fastmcp_init.side_effect = capture_instance

    mcp = MCP(rag_manager=mock_rag_manager, kb_manager=mock_kb_manager)

    # Verify FastMCP instantiation
    expected_name = "Knowledge Base MCP"
    expected_instructions = dedent("""
            Tools to query multiple custom knowledge bases using similarity search and a ranked knowledge-graph. 
            Search modes: 
            - local: Focuses on context-dependent information.
            - global: Utilizes global knowledge.
            - hybrid: Combines local and global retrieval methods.
            - naive: Performs a basic search without advanced techniques.
            - mix: Integrates knowledge graph and vector retrieval.
            """)
    mock_fastmcp_init.assert_called_once_with(name=expected_name, instructions=expected_instructions)
    
    # Verify FastMCP.run call
    mock_fastmcp_run.assert_called_once_with(transport="stdio")

    # Verify tools were added
    # Access the captured FastMCP instance to check calls to its add_tool method
    captured_fmc_instance = real_fastmcp_instance_storage.get('instance')
    assert captured_fmc_instance is not None
    
    calls = captured_fmc_instance.add_tool.call_args_list
    assert len(calls) == 3

    # Check retrieve tool
    retrieve_call = next(c for c in calls if c[1]['name'] == 'retrieve')
    assert retrieve_call[0][0] == mcp.retrieve # Tool function
    assert "Returns the retrieval results only." in retrieve_call[1]['description']

    # Check answer tool
    answer_call = next(c for c in calls if c[1]['name'] == 'answer')
    assert answer_call[0][0] == mcp.answer
    assert "Returns an LLM-synthesised answer" in answer_call[1]['description']
    
    # Check list_knowledgebases tool
    list_kb_call = next(c for c in calls if c[1]['name'] == 'list_knowledgebases')
    assert list_kb_call[0][0] == mcp.list_knowledgebases
    assert "List all available knowledge bases." in list_kb_call[1]['description']

@patch('knowledge_mcp.mcp_server.FastMCP.run')
def test_mcp_init_invalid_rag_manager_type(mock_run, mock_kb_manager):
    """Test TypeError if rag_manager is not a RagManager instance."""
    with pytest.raises(TypeError, match="Invalid RagManager instance provided"):
        MCP(rag_manager="not_a_rag_manager", kb_manager=mock_kb_manager)

@patch('knowledge_mcp.mcp_server.FastMCP.run')
def test_mcp_init_invalid_kb_manager_type(mock_run, mock_rag_manager):
    """Test TypeError if kb_manager is not a KnowledgeBaseManager instance."""
    with pytest.raises(TypeError, match="Invalid KnowledgeBaseManager instance provided"):
        MCP(rag_manager=mock_rag_manager, kb_manager="not_a_kb_manager")


# 2. retrieve tool (async method)
@pytest.mark.asyncio
async def test_retrieve_tool_success(mcp_instance, mock_rag_manager):
    """Test successful call to retrieve tool."""
    kb_name = "test_kb"
    query_text = "What is test?"
    mode = "mix"
    top_k = 50
    ids = ["doc1", "doc2"]
    
    mock_rag_manager.query.return_value = "Retrieved context"

    result = await mcp_instance.retrieve(kb=kb_name, query=query_text, mode=mode, top_k=top_k, ids=ids)

    mock_rag_manager.query.assert_awaited_once_with(
        kb_name=kb_name,
        query_text=query_text,
        mode=mode,
        top_k=top_k,
        ids=ids,
        only_need_context=True
    )
    assert result == "Retrieved context" # _wrap_result just calls str()

@pytest.mark.asyncio
async def test_retrieve_tool_error_kb_not_found(mcp_instance, mock_rag_manager):
    """Test retrieve tool converts KnowledgeBaseNotFoundError to ValueError."""
    mock_rag_manager.query.side_effect = KnowledgeBaseNotFoundError("KB not found")
    with pytest.raises(ValueError, match="KB not found"):
        await mcp_instance.retrieve(kb="any_kb", query="any_query", mode="mix", top_k=30, ids=None)

@pytest.mark.asyncio
async def test_retrieve_tool_error_config_error(mcp_instance, mock_rag_manager):
    """Test retrieve tool converts ConfigurationError to ValueError."""
    mock_rag_manager.query.side_effect = ConfigurationError("Bad config")
    with pytest.raises(ValueError, match="Bad config"):
        await mcp_instance.retrieve(kb="any_kb", query="any_query", mode="mix", top_k=30, ids=None)

@pytest.mark.asyncio
async def test_retrieve_tool_error_ragmanager_error(mcp_instance, mock_rag_manager):
    """Test retrieve tool converts RAGManagerError to RuntimeError."""
    mock_rag_manager.query.side_effect = RAGManagerError("RAG failed")
    with pytest.raises(RuntimeError, match="Query failed: RAG failed"):
        await mcp_instance.retrieve(kb="any_kb", query="any_query", mode="mix", top_k=30, ids=None)

@pytest.mark.asyncio
async def test_retrieve_tool_error_unexpected_exception(mcp_instance, mock_rag_manager):
    """Test retrieve tool converts unexpected Exception to RuntimeError."""
    mock_rag_manager.query.side_effect = Exception("Something broke")
    with pytest.raises(RuntimeError, match="An unexpected error occurred: Something broke"):
        await mcp_instance.retrieve(kb="any_kb", query="any_query", mode="mix", top_k=30, ids=None)

# 3. answer tool (async method)
@pytest.mark.asyncio
async def test_answer_tool_success(mcp_instance, mock_rag_manager):
    """Test successful call to answer tool."""
    kb_name = "test_kb_ans"
    query_text = "Explain test."
    mode = "hybrid"
    top_k = 20
    response_type = "Single Paragraph"
    ids = None
    
    mock_rag_manager.query.return_value = "Synthesized answer."

    result = await mcp_instance.answer(
        kb=kb_name, query=query_text, mode=mode, top_k=top_k, response_type=response_type, ids=ids
    )

    mock_rag_manager.query.assert_awaited_once_with(
        kb_name=kb_name,
        query_text=query_text,
        mode=mode,
        top_k=top_k,
        response_type=response_type,
        ids=ids,
        only_need_context=False
    )
    assert result == "Synthesized answer."

@pytest.mark.asyncio
async def test_answer_tool_error_kb_not_found(mcp_instance, mock_rag_manager):
    """Test answer tool converts KnowledgeBaseNotFoundError to ValueError."""
    mock_rag_manager.query.side_effect = KnowledgeBaseNotFoundError("KB not found for answer")
    with pytest.raises(ValueError, match="KB not found for answer"):
        await mcp_instance.answer(kb="any_kb", query="any", mode="mix", top_k=30, response_type="P", ids=None)

@pytest.mark.asyncio
async def test_answer_tool_error_config_error(mcp_instance, mock_rag_manager):
    """Test answer tool converts ConfigurationError to ValueError."""
    mock_rag_manager.query.side_effect = ConfigurationError("Bad config for answer")
    with pytest.raises(ValueError, match="Bad config for answer"):
        await mcp_instance.answer(kb="any_kb", query="any", mode="mix", top_k=30, response_type="P", ids=None)

@pytest.mark.asyncio
async def test_answer_tool_error_ragmanager_error(mcp_instance, mock_rag_manager):
    """Test answer tool converts RAGManagerError to RuntimeError."""
    mock_rag_manager.query.side_effect = RAGManagerError("RAG failed for answer")
    with pytest.raises(RuntimeError, match="Query failed: RAG failed for answer"):
        await mcp_instance.answer(kb="any_kb", query="any", mode="mix", top_k=30, response_type="P", ids=None)

@pytest.mark.asyncio
async def test_answer_tool_error_unexpected_exception(mcp_instance, mock_rag_manager):
    """Test answer tool converts unexpected Exception to RuntimeError."""
    mock_rag_manager.query.side_effect = Exception("Something broke in answer")
    with pytest.raises(RuntimeError, match="An unexpected error occurred: Something broke in answer"):
        await mcp_instance.answer(kb="any_kb", query="any", mode="mix", top_k=30, response_type="P", ids=None)

# 4. list_knowledgebases tool (async method)
@pytest.mark.asyncio
async def test_list_knowledgebases_tool_success(mcp_instance, mock_kb_manager):
    """Test successful call to list_knowledgebases tool."""
    kb_data = {"kb1": "Description for KB1", "kb2": "Description for KB2"}
    mock_kb_manager.list_kbs.return_value = kb_data

    result_str = await mcp_instance.list_knowledgebases()
    
    mock_kb_manager.list_kbs.assert_awaited_once()
    
    expected_output = {
        "knowledge_bases": [
            {"name": "kb1", "description": "Description for KB1"},
            {"name": "kb2", "description": "Description for KB2"}
        ]
    }
    assert json.loads(result_str) == expected_output

@pytest.mark.asyncio
async def test_list_knowledgebases_tool_empty(mcp_instance, mock_kb_manager):
    """Test list_knowledgebases tool when no KBs exist."""
    mock_kb_manager.list_kbs.return_value = {}
    result_str = await mcp_instance.list_knowledgebases()
    expected_output = {"knowledge_bases": []}
    assert json.loads(result_str) == expected_output

@pytest.mark.asyncio
async def test_list_knowledgebases_tool_error_kb_error(mcp_instance, mock_kb_manager):
    """Test list_knowledgebases tool converts KnowledgeBaseError to ValueError."""
    mock_kb_manager.list_kbs.side_effect = KnowledgeBaseError("Listing failed")
    with pytest.raises(ValueError, match="Failed to list knowledge bases: Listing failed"):
        await mcp_instance.list_knowledgebases()

@pytest.mark.asyncio
async def test_list_knowledgebases_tool_error_unexpected(mcp_instance, mock_kb_manager):
    """Test list_knowledgebases tool converts unexpected Exception to RuntimeError."""
    mock_kb_manager.list_kbs.side_effect = Exception("System broke")
    with pytest.raises(RuntimeError, match="An unexpected server error occurred: System broke"):
        await mcp_instance.list_knowledgebases()

"""
This test suite covers the MCP class from knowledge_mcp.mcp_server.

Key areas tested:
- Initialization (__init__):
    - Successful initialization with mocked RagManager and KnowledgeBaseManager.
    - TypeError for invalid manager types.
    - Verification of FastMCP instantiation (name, instructions) by patching its __init__.
    - Verification of tool addition (retrieve, answer, list_knowledgebases) with correct details by checking calls to a mocked add_tool on the captured FastMCP instance.
    - Ensured FastMCP.run is called with transport="stdio" (by patching it).
- retrieve tool (async):
    - Successful call, checking parameters passed to RagManager.query (especially only_need_context=True) and result wrapping.
    - Error Handling: Conversion of KnowledgeBaseNotFoundError/ConfigurationError to ValueError, and RAGManagerError/Exception to RuntimeError.
- answer tool (async):
    - Successful call, checking parameters passed to RagManager.query (especially only_need_context=False) and result wrapping.
    - Error Handling: Similar to retrieve, ensuring correct error type conversions.
- list_knowledgebases tool (async):
    - Successful call, verifying transformation of KnowledgeBaseManager.list_kbs output (dict) to the specified JSON string format.
    - Error Handling: Conversion of KnowledgeBaseError to ValueError, and unexpected Exception to RuntimeError.

Utilized pytest.mark.asyncio, unittest.mock (MagicMock, AsyncMock, patch),
and fixtures for dependencies. The FastMCP.run method is patched in a fixture for
most tests or directly in specific initialization tests to prevent actual execution.
"""
