# Product Requirements Document: knowledge-mcp

## 1. Overview and Objectives

**knowledge-mcp** is a Python-based tool that provides searchable knowledge bases through an MCP server interface, along with a CLI for knowledge base management. The primary purpose is to enable AI assistants to proactively query specialized knowledge bases during their reasoning process, rather than relying solely on semantic search against the user's initial prompt.

### Key Objectives:
- Provide a simple CLI interface for knowledge base management
- Implement an MCP server that exposes search functionality
- Support multiple document formats (PDF, text, markdown, doc)
- Enable AI assistants to search knowledge bases during their chain-of-thought reasoning

## 2. Technical Stack

- **Language:** Python 3.12
- **Dependency Management:** uv
- **Knowledge Base Technology:** LightRAG (https://github.com/HKUDS/LightRAG)
- **MCP Server Implementation:** FastMCP (https://github.com/jlowin/fastmcp)
- **Model Provider:** OpenAI (for MVP)

## 3. Core Functionality

### 3.1 CLI Tool

The CLI tool provides the following commands:

| Command | Description | Arguments | Status |
|---------|-------------|-----------|--------|
| `create` | Creates a new knowledge base | `<kb-name>`: Name of the knowledge base to create | Implemented |
| `delete` | Deletes an existing knowledge base | `<kb-name>`: Name of the knowledge base to delete | Implemented |
| `list`   | Lists all available knowledge bases | N/A | Implemented |
| `add`    | Adds a document to a knowledge base (processes and embeds) | `<kb-name>`: Target knowledge base<br>`<path>`: Path to the document | Implemented |
| `remove` | Removes a document from a knowledge base (removes embeddings) | `<kb-name>`: Target knowledge base<br>`<doc_name>`: Name of the document to remove | Implemented |
| `config` | Manage KB-specific config | `<kb_name>`: Target KB<br>`[show|edit]`: Subcommand (show default) | Implemented |
| `search` | Searches the knowledge base | `<kb-name>`: Target knowledge base<br>`<query>`: Search query | Implemented |
| `mcp`    | Runs the MCP server | N/A | Pending |

**Required option for all commands:**
- `--config`: Path to the configuration file (mandatory)

### 3.2 MCP Server

(Status: Pending)

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

(Status: Main configuration implemented)

The configuration file (YAML format) contains the following sections:

```yaml
knowledge_base:
  base_dir: .

lightrag:
  llm:
    provider: "openai"
    model_name: "gpt-4.1-nano"
    max_token_size: 32768
    api_key: "${OPENAI_API_KEY}"
    api_base: "${OPENAI_API_BASE}"
    kwargs:
      temperature: 0.0
      top_p: 0.9
  embedding:
    provider: "openai"
    model_name: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    api_base: "${OPENAI_API_BASE}"
    embedding_dim: 1536
    max_token_size: 8191
  embedding_cache:
    enabled: true
    similarity_threshold: 0.90

logging:
  level: "INFO"
  max_bytes: 10485760
  backup_count: 5
  detailed_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  default_format: "%(levelname)s: %(message)s"

env_file: .env
```

(Status: KB-specific configuration schema defined, generated on KB creation. Integration pending)

Each knowledge base has its own configuration file in the `<base_dir>/<kb_name>/` directory.
```yaml
history_turns: 3
max_token_for_global_context: 4000
max_token_for_local_context: 4000
max_token_for_text_unit: 4000
mode: mix
only_need_context: false
only_need_prompt: false
response_type: Multiple Paragraphs
top_k: 60
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
├── .env.example
├── .gitignore
├── .python-version
├── .windsurfrules
├── README.md
├── README-task-master.md
├── _project/                 # Contains PRD, plan etc.
│   └── prd.md
├── config.example.yaml
├── kbs/                      # Default base directory for knowledge bases
├── knowledge_mcp/          # Source code
│   ├── __init__.py
│   ├── cli.py                # Main CLI entrypoint (using shell)
│   ├── config.py             # Global configuration handling
│   ├── documents.py          # Document processing logic
│   ├── knowledgebases.py     # KB directory management
│   ├── mcp_server.py         # MCP server implementation (pending)
│   ├── rag.py                # LightRAG integration and search
│   ├── shell.py              # Interactive shell implementation
│   └── utils.py              # Utility functions (if any)
├── pyproject.toml            # Project metadata and dependencies (uv/hatch)
├── sample_doc.txt
├── scripts/                  # Utility scripts (like task management)
├── tasks/                    # Task management files (JSON, markdown)
├── tests/                    # Test suite
└── uv.lock
```

## 7. Development Roadmap

### Phase 1: Core Infrastructure (Completed)
- Set up project structure with Python 3.12 and uv
- Implement configuration file parsing
- Create basic CLI command structure
- Implement knowledge base directory creation/deletion

### Phase 2: Document Management (Completed)
- Implement document addition functionality (Embedding/storage pending) (Implemented via LightRAG)
- Integrate with LightRAG for document processing (Embedding/storage pending) (Implemented)
- Implement document removal functionality (Embedding/storage pending) (Implemented via LightRAG)
- Add support for different document types (PDF, text, markdown, doc)

### Phase 3: Search Functionality (Pending)
- Implement search functionality using LightRAG
- Set up proper result formatting
- Add basic logging

### Phase 4: MCP Server (Pending)
- Integrate FastMCP
- Implement the search method
- Set up server configuration
- Test with sample MCP clients

### Phase 5: Refinement and Testing (Pending)
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
- Web interface for knowledge base management, e.g. with the LightRAG web-ui
- Support for document updates without delete/re-add
- Performance optimizations for large knowledge bases
