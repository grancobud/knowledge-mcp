# Task List: Alternative Document Parsing Methods

## Relevant Files

- `knowledge_mcp/documents.py` - Refactor to replace Textract with Markitdown for text extraction
- `knowledge_mcp/shell.py` - Add new `add-multimodal` and `add-text` commands and integrate DocumentManager
- `knowledge_mcp/cli.py` - Update CLI interface to support `--method` option for parsing method selection
- `knowledge_mcp/rag.py` - Ensure RagManager can provide access to underlying LightRAG instances for text-only processing
- `pyproject.toml` - Update dependencies to add Markitdown and remove Textract
- `requirements.txt` (if exists) - Update dependencies accordingly

### Notes

- The implementation focuses on providing two distinct parsing paths: multimodal (RagAnything) and text-only (Markitdown + LightRAG)
- Existing knowledge bases should continue to work without modification
- Mixed-mode knowledge bases (containing documents processed with different methods) should be supported
- Use the context7 MCP tool to look up the latest documentation for LightRAG and Markitdown libraries when implementing

## Tasks

- [x] 1.0 Refactor DocumentManager for Markitdown Integration
  - [x] 1.1 Remove all Textract-related imports and code from documents.py
  - [x] 1.2 Add Markitdown import and initialize Markitdown instance
  - [x] 1.3 Replace _extract_text method to use Markitdown instead of Textract
  - [x] 1.4 Update SUPPORTED_EXTENSIONS list to match Markitdown capabilities
  - [x] 1.5 Update error handling for Markitdown-specific exceptions
  - [x] 1.6 Add logging for Markitdown text extraction process
  - [x] 1.7 Test DocumentManager with various file types using Markitdown

- [x] 2.0 Enhance RagManager for Direct LightRAG Access (COMPLETED - Supporting Infrastructure)
  - [x] 2.1 Add method to get underlying LightRAG instance from RAGAnything
  - [x] 2.2 Create text-only ingestion method that bypasses RAGAnything
  - [x] 2.3 Add method parameter to ingest_document for parsing method selection
  - [x] 2.4 Implement logic to route to appropriate ingestion method based on parse_method
  - [x] 2.5 Update logging to indicate which parsing method was used
  - [x] 2.6 Ensure backward compatibility with existing ingest_document calls

- [x] 2.1 Enhance DocumentManager with Dual Parsing Methods
  - [x] 2.1.1 Rename existing add method to add_multimodal (maintains current multimodal behavior)
  - [x] 2.1.2 Create add_text_only method that uses MarkItDown + RagManager.ingest_text_only
  - [x] 2.1.3 Create generic add method with method parameter for routing between the two
  - [x] 2.1.4 Add comprehensive logging to distinguish between parsing methods
  - [x] 2.1.5 Ensure backward compatibility for existing add method calls
  - [x] 2.1.6 Add proper error handling for both parsing methods

- [ ] 3.0 Implement New Shell Commands
  - [x] 3.1 Uncomment and update DocumentManager import in shell.py
  - [x] 3.2 Initialize DocumentManager instance in Shell.__init__
  - [x] 3.3 Implement do_add_multimodal command (calls DocumentManager.add_multimodal)
  - [x] 3.4 Implement do_add_text command (calls DocumentManager.add_text_only)
  - [x] 3.5 Add help text for both new commands
  - [x] 3.6 Update existing do_add command to maintain backward compatibility
  - [x] 3.7 Add error handling specific to each parsing method
  - [ ] 3.8 Test both commands with various document types

- [ ] 4.0 Update CLI Interface for Method Selection
  - [ ] 4.1 Add --method argument to CLI add command parser
  - [ ] 4.2 Define valid method choices (multimodal, text)
  - [ ] 4.3 Set default method to multimodal for backward compatibility
  - [ ] 4.4 Pass method parameter to underlying ingestion functions
  - [ ] 4.5 Update CLI help text to document method options
  - [ ] 4.6 Add validation for method parameter values
  - [ ] 4.7 Test CLI with both method options
