# Product Requirements Document: knowledge-mcp

## 1. Overview and Objectives

**knowledge-mcp** is a Python-based tool that provides searchable knowledge bases through an MCP server interface, along with a CLI for knowledge base management. The primary purpose is to enable AI assistants to proactively query specialized knowledge bases during their reasoning process, rather than relying solely on semantic search against the user's initial prompt.

### Key Objectives:
- Provide a simple CLI interface for knowledge base management
- Implement an MCP server that exposes search functionality
- Support multiple document formats (PDF, text, markdown, doc)
- Enable AI assistants to search knowledge bases during their chain-of-thought reasoning

## 2. Technical Stack

- **Language:** Python 3.13
- **Dependency Management:** uv
- **Knowledge Base Technology:** LightRAG (https://github.com/HKUDS/LightRAG)
- **MCP Server Implementation:** FastMCP (https://github.com/jlowin/fastmcp)
- **Model Provider:** OpenAI (for MVP)

## 3. Core Functionality

### 3.1 CLI Tool

The CLI tool provides the following commands:

| Command | Description | Arguments |
|---------|-------------|-----------|
| `create` | Creates a new knowledge base | `<kb-name>`: Name of the knowledge base to create |
| `delete` | Deletes an existing knowledge base | `<kb-name>`: Name of the knowledge base to delete |
| `add` | Adds a document to a knowledge base | `<kb-name>`: Target knowledge base<br>`<path>`: Path to the document<br>`<name>`: Name to assign to the document |
| `remove` | Removes a document from a knowledge base | `<kb-name>`: Target knowledge base<br>`<name>`: Name of the document to remove |
| `search` | Searches the knowledge base | `<kb-name>`: Target knowledge base<br>`<query>`: Search query |
| `mcp` | Runs the MCP server | N/A |

**Required option for all commands:**
- `--config`: Path to the configuration file (mandatory)

### 3.2 MCP Server

The MCP server exposes the following method:
- `search <kb-name> <query>`: Searches the specified knowledge base with the given query

**Example MCP configuration:**
```json
{
  "mcpServers": {
    "knowledge-mcp": {
      "command": "uvx",
      "args": [
        "knowledge-mcp",
        "mcp",
        "--config", 
        "/path/to/knowledge-mcp.yaml"
      ]
    }
  }
}
```

## 4. Configuration

The configuration file (YAML format) contains the following sections:

```yaml
# Example configuration
knowledge_base:
  base_dir: "/path/to/kb/storage"
  
embedding_model:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  model_name: "text-embedding-3-small"
  api_key: "${OPENAI_API_KEY}"  # Environment variable support

language_model:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  model_name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"

logging:
  level: "INFO"
  file: "/path/to/log/file.log"
```

## 5. Implementation Details

### 5.1 Knowledge Base Structure

- Each knowledge base is stored in its own directory: `<base_dir>/<kb_name>/`
- Documents are stored with vector embeddings in the knowledge base directory
- Document updates require deletion and re-ingestion

### 5.2 Document Processing

- Supported formats: PDF, text, markdown, doc
- Document processing leverages LightRAG's default chunking strategy
- Each document is processed, chunked, and stored with vector embeddings

### 5.3 Search Implementation

- Uses LightRAG's in-context mode for searches
- Returns relevant text chunks and entities from the knowledge graph
- Search results format is determined by LightRAG's in-context mode output

### 5.4 Error Handling

- Informative error messages for common failure scenarios
- Proper exit codes for CLI commands
- Validation of configuration and input parameters

### 5.5 Logging

- Simple logging mechanism configured in the YAML file
- Logs operations and errors for debugging

## 6. Project Structure

```
knowledge-mcp/
├── pyproject.toml            # Project metadata and dependencies
├── knowledge_mcp/
│   ├── __init__.py
│   ├── cli.py                # CLI implementation
│   ├── config.py             # Configuration handling
│   ├── kb_manager.py         # Knowledge base management
│   ├── document_processor.py # Document processing and embedding
│   ├── search.py             # Search functionality
│   ├── mcp_server.py         # MCP server implementation
│   └── utils.py              # Utility functions
├── tests/                    # Test suite
├── docs/                     # Documentation
└── examples/                 # Example configurations and usage
```

## 7. Development Roadmap

### Phase 1: Core Infrastructure
- Set up project structure with Python 3.13 and uv
- Implement configuration file parsing
- Create basic CLI command structure
- Implement knowledge base directory creation/deletion

### Phase 2: Document Management
- Implement document addition functionality
- Integrate with LightRAG for document processing
- Implement document removal functionality
- Add support for different document types (PDF, text, markdown, doc)

### Phase 3: Search Functionality
- Implement search functionality using LightRAG
- Set up proper result formatting
- Add basic logging

### Phase 4: MCP Server
- Integrate FastMCP
- Implement the search method
- Set up server configuration
- Test with sample MCP clients

### Phase 5: Refinement and Testing
- Comprehensive error handling
- Optimization of search performance
- Documentation
- End-to-end testing

## 8. Technical Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Handling various document formats | Use specialized libraries for each format (PyPDF2 for PDFs, python-docx for doc files) |
| Managing API costs for embeddings | Implement caching and batching strategies |
| LightRAG integration | Thorough testing and potentially contributing improvements to the project |
| MCP protocol compatibility | Use FastMCP library and test with different clients |
| API key security | Support environment variable substitution in config files |
| Error handling | Implement proper retry mechanisms and fallbacks |
| Performance with large knowledge bases | Optimize vector storage and retrieval operations |

## 9. Future Enhancements (Post-MVP)

- Support for additional model providers beyond OpenAI
- Custom chunking strategies for document processing
- Authentication for the MCP server
- Web interface for knowledge base management
- Support for document updates without delete/re-add
- Performance optimizations for large knowledge bases
