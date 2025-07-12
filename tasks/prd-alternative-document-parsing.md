# Product Requirements Document: Alternative Document Parsing Methods

## Introduction/Overview

This feature adds alternative document parsing methods to the Knowledge MCP system to provide users with more flexibility in how documents are processed and ingested into knowledge bases. Currently, the system uses RagAnything for multimodal document processing, but users need the ability to choose between multimodal parsing (with RagAnything) and text-only parsing (using Markitdown) based on their specific use cases.

The feature addresses the need for:
- Lightweight text-only document processing for scenarios where multimodal features aren't needed
- Replacing the deprecated Textract dependency that conflicts with RagAnything
- Providing users explicit control over parsing methods during document ingestion

## Goals

1. **Provide parsing method choice**: Enable users to explicitly choose between multimodal and text-only document parsing
2. **Replace Textract dependency**: Eliminate dependency conflicts by replacing Textract with Markitdown for text extraction
3. **Maintain backward compatibility**: Ensure existing knowledge bases continue to work without modification
4. **Improve user control**: Give users explicit control over document processing methods
5. **Enable mixed-mode knowledge bases**: Support knowledge bases with documents processed using different parsing methods

## User Stories

1. **As a developer**, I want to ingest documents using text-only parsing so that I can create lightweight knowledge bases without multimodal overhead.

2. **As a researcher**, I want to choose multimodal parsing for documents with images and charts so that I can capture all visual information in my knowledge base.

3. **As a system administrator**, I want to avoid dependency conflicts so that I can maintain a stable deployment without Textract compatibility issues.

4. **As a CLI user**, I want to specify parsing methods as command options so that I can integrate document ingestion into automated workflows.

5. **As a shell user**, I want separate commands for different parsing methods so that I can quickly choose the appropriate method interactively.

## Functional Requirements

1. **Shell Interface Requirements**:
   - The system must provide an `add-multimodal` command that uses RagAnything for full multimodal document processing
   - The system must provide an `add-text` command that uses Markitdown for text-only extraction and LightRAG for ingestion
   - Both commands must follow the syntax: `<command> <kb_name> <file_path>`
   - The existing `add` command must continue to work for backward compatibility

2. **CLI Interface Requirements**:
   - The CLI must support a `--method` or similar option to specify parsing method
   - Valid method options must include `multimodal` and `text`
   - The CLI must default to multimodal parsing if no method is specified

3. **Document Processing Requirements**:
   - The `add-text` command must use Markitdown to extract text content from documents
   - The `add-text` command must use the underlying LightRAG instance directly (bypassing RagAnything)
   - The `add-multimodal` command must continue using RagAnything as currently implemented
   - The system must completely replace Textract functionality with Markitdown

4. **DocumentManager Integration**:
   - The obsolete `documents.py` file must be refactored to use Markitdown instead of Textract
   - The DocumentManager class must be reintegrated into the shell workflow for text-only processing
   - The DocumentManager must not maintain Textract as an option

5. **Logging and Feedback**:
   - The system must log which parsing method was used for each document ingestion
   - Users must receive confirmation of successful document addition with the method used
   - Error messages must clearly indicate parsing method-related failures

6. **Knowledge Base Compatibility**:
   - Existing knowledge bases must continue to function without modification
   - Knowledge bases must support mixed-mode content (documents processed with different parsing methods)
   - The ingestion method must not affect query operations on existing content

## Non-Goals (Out of Scope)

1. **Automatic parsing method detection** - Users must explicitly specify the parsing method
2. **Fallback mechanisms** - No automatic fallback to alternative parsing methods on failure  
3. **Migration tools** - No tools to convert existing knowledge bases between parsing methods
4. **Per-knowledge-base default preferences** - No persistent parsing method preferences
5. **Validation of parsing method compatibility** - No prevention of mixed-mode operations
6. **Textract maintenance** - Complete removal of Textract dependency and functionality

## Technical Considerations

1. **Architecture Changes**:
   - Refactor `documents.py` to replace Textract with Markitdown
   - Modify `shell.py` to add new commands and integrate DocumentManager
   - Update CLI interface to support method selection options
   - Ensure RagManager can provide access to underlying LightRAG instances

2. **Dependencies**:
   - Add Markitdown as a new dependency
   - Remove Textract dependency completely
   - Ensure no conflicts between Markitdown and existing RagAnything dependencies

3. **Data Flow**:
   - `add-multimodal`: File → RagAnything → Knowledge Base
   - `add-text`: File → Markitdown → Text → LightRAG → Knowledge Base

4. **Error Handling**:
   - Handle Markitdown parsing failures gracefully
   - Provide clear error messages for unsupported file types
   - Log parsing method and success/failure status

## Design Considerations

1. **Command Interface**:
   - Shell commands should be intuitive and self-documenting
   - CLI options should follow standard conventions
   - Help text should clearly explain the differences between parsing methods

2. **User Experience**:
   - Commands should provide immediate feedback about parsing method selection
   - Error messages should guide users toward appropriate parsing methods
   - Documentation should explain when to use each parsing method

## Success Metrics

1. **Functionality**: Users can successfully ingest documents using both `add-multimodal` and `add-text` commands
2. **Dependency Resolution**: Complete removal of Textract dependency conflicts
3. **Performance**: Text-only parsing shows improved performance for text documents compared to multimodal parsing
4. **Compatibility**: Existing knowledge bases continue to function without modification
5. **User Adoption**: Users actively choose appropriate parsing methods based on their use cases

## Open Questions

1. Should there be any file type restrictions for each parsing method?
2. Should the system provide guidance on which parsing method to use for specific file types?
3. How should the system handle edge cases where Markitdown fails to extract text from a document?
4. Should there be any performance monitoring or metrics collection for different parsing methods?
5. Are there any specific Markitdown configuration options that should be exposed to users?
