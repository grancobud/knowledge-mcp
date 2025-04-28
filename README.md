# knowledge-mcp: Specialized Knowledge Bases for AI Agents

## 1. Overview and Concept

**knowledge-mcp** is a MCP server designed to bridge the gap between specialized knowledge domains and AI assistants. It allows users to create, manage, and query dedicated knowledge bases, making this information accessible to AI agents through an MCP (Model Context Protocol) server interface.

The core idea is to empower AI assistants that are MCP clients (like Claude Desktop or IDEs like Windsurf) to proactively consult these specialized knowledge bases during their reasoning process (Chain of Thought), rather than relying solely on general semantic search against user prompts or broad web searches. This enables more accurate, context-aware responses when dealing with specific domains.

Key components:

*   **CLI Tool:** Provides a user-friendly command-line interface for managing knowledge bases (creating, deleting, adding/removing documents, configuring, searching).
*   **Knowledge Base Engine:** Leverages **LightRAG** to handle document processing, embedding, knowledge graph creation, and complex querying.
*   **MCP Server:** Exposes the search functionality of the knowledge bases via the FastMCP protocol, allowing compatible AI agents to query them directly.

## 2. About LightRAG

This project utilizes LightRAG ([HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)) as its core engine for knowledge base creation and querying. LightRAG is a powerful framework designed to enhance Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG) with knowledge graph techniques.

Key features of LightRAG relevant to this project:

*   **Document Processing Pipeline:** Ingests documents (PDF, Text, Markdown, DOCX), chunks them, extracts entities and relationships using an LLM, and builds both a knowledge graph and vector embeddings.
*   **Multiple Query Modes:** Supports various retrieval strategies (e.g., vector similarity, entity-centric, relationship-focused, hybrid) to find the most relevant context for a given query.
*   **Flexible Storage:** Can use different backends for storing key-value data, vectors, graph information, and document status (this project uses the default file-based storage).
*   **LLM/Embedding Integration:** Supports various providers like OpenAI (used in this project), Ollama, Hugging Face, etc.

By using LightRAG, `knowledge-mcp` benefits from advanced RAG capabilities that go beyond simple vector search.

## 3. Installation

Ensure you have Python 3.12 and `uv` installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/olafgeibig/knowledge-mcp.git
    cd knowledge-mcp
    ```

2.  **Create a virtual environment and install dependencies using uv:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # Or .\.venv\Scripts\activate on Windows
    uv pip install -e ".[dev]"
    ```
    *Installing with `-e .` makes the package editable and installs dev dependencies.* 

3.  **Set up configuration:**
    *   Copy `config.example.yaml` to `config.yaml`.
    *   Copy `.env.example` to `.env`.
    *   Edit `config.yaml` and `.env` to add your API keys (e.g., `OPENAI_API_KEY`) and adjust paths or settings as needed. The `knowledge_base.base_dir` in `config.yaml` specifies where your knowledge base directories will be created.

## 4. Usage (CLI)

The primary way to interact with `knowledge-mcp` is through its CLI, accessed via the `knowledge-mcp` command (if installed globally or via `uvx knowledge-mcp` within the activated venv).

**All commands require the `--config` option pointing to your main configuration file.**

```bash
knowledge-mcp --config config.yaml <command> [arguments...]
```

**Available Commands:**

| Command  | Description                                                                 | Arguments                                                                    | Status      |
| :------- | :-------------------------------------------------------------------------- | :--------------------------------------------------------------------------- | :---------- |
| `create` | Creates a new knowledge base directory and initializes its structure.       | `<kb-name>`: Name of the knowledge base to create.                           | Implemented |
| `delete` | Deletes an existing knowledge base directory and all its contents.            | `<kb-name>`: Name of the knowledge base to delete.                           | Implemented |
| `list`   | Lists all available knowledge bases found in the `base_dir`.                | N/A                                                                          | Implemented |
| `add`    | Adds a document: processes, chunks, embeds, stores in the specified KB.     | `<kb-name>`: Target KB.<br>`<path>`: Path to the document file.             | Implemented |
| `remove` | Removes a document and its associated embeddings from the KB.             | `<kb-name>`: Target KB.<br>`<doc_name>`: Name/ID of the document to remove. | Implemented |
| `config` | Manages the KB-specific `config.yaml` (query parameters).                 | `<kb_name>`: Target KB.<br>`[show|edit]`: Subcommand (show default).        | Implemented |
| `search` | Searches the specified knowledge base using LightRAG.                     | `<kb-name>`: Target KB.<br>`<query>`: Your search query text.              | Implemented |
| `mcp`    | Runs the MCP server to expose the `search` functionality to AI agents.      | N/A                                                                          | Pending     |
| `shell`  | Starts an interactive shell session with all commands available.            | N/A                                                                          | Implemented |
| `exit`   | (Within shell) Exits the interactive shell.                                 | N/A                                                                          | Implemented |
| `help`   | (Within shell) Shows available commands and their usage.                    | `[command]` (Optional command name)                                          | Implemented |

**Example:**

```bash
# Create a knowledge base named 'my_docs'
knowledge-mcp --config config.yaml create my_docs

# Add a document to it
knowledge-mcp --config config.yaml add my_docs ./path/to/mydocument.pdf

# Search the knowledge base
knowledge-mcp --config config.yaml search my_docs "What is the main topic?"

# Start the interactive shell
knowledge-mcp --config config.yaml shell

(kbmcp) list
(kbmcp) search my_docs "Another query"
(kbmcp) exit
```

## 5. Configuration

Configuration is managed via YAML files:

*   **Main Configuration (`config.yaml`):** Defines global settings like the knowledge base directory (`knowledge_base.base_dir`), LightRAG parameters (LLM provider/model, embedding provider/model, API keys via `${ENV_VAR}` substitution), and logging settings.

    ```yaml
    # Example structure (see config.example.yaml for full details)
    knowledge_base:
      base_dir: ./kbs # Default directory for KBs

    lightrag:
      llm:
        provider: "openai"
        model_name: "gpt-4.1-nano"
        api_key: "${OPENAI_API_KEY}"
        # ... other LLM settings
      embedding:
        provider: "openai"
        model_name: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        # ... other embedding settings
      embedding_cache:
        enabled: true
        similarity_threshold: 0.90

    logging:
      level: "INFO"
      # ... logging settings

    env_file: .env # Optional path to .env file
    ```

*   **Knowledge Base Specific Configuration (`<base_dir>/<kb_name>/config.yaml`):** Contains parameters specific to querying *that* knowledge base, such as the LightRAG query `mode`, `top_k` results, context token limits, etc. This file is automatically created with defaults when a KB is created and can be viewed/edited using the `config` CLI command.

## 6. Development

*   **Tech Stack:** Python 3.12, uv (dependency management), hatchling (build system), pytest (testing).
*   **Setup:** Follow the installation steps, ensuring you install with `uv pip install -e ".[dev]"`.
*   **Code Style:** Adheres to PEP 8.
*   **Testing:** Run tests using `uvx test` or `pytest`.
*   **Dependencies:** Managed in `pyproject.toml`. Use `uv pip install <package>` to add and `uv pip uninstall <package>` to remove dependencies, updating `pyproject.toml` accordingly.
*   **Scripts:** Common tasks might be defined under `[project.scripts]` in `pyproject.toml`.
*   **MCP Inspector:** Use `npx @modelcontextprotocol/inspector uv run cli --config ./kbs/config.yaml serve` to start the MCP inspector.
