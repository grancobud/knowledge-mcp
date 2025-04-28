# knowledge_mcp/mcp_server.py
"""FastMCP server exposing tools to interact with knowledge bases."""

import logging
# import asyncio # Removed unused import
import json
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field, field_validator
from fastmcp import FastMCP

# Import necessary exceptions and manager types
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError, KnowledgeBaseError # Added KbManager
from knowledge_mcp.rag import ConfigurationError, RAGManagerError, RagManager

logger = logging.getLogger(__name__)

# --- Helper Function ---
def _wrap_result(result: Any) -> str:
    """Simple wrapper to ensure string output, can be enhanced."""
    return str(result)

# --- Pydantic Models for Tool Parameters (remain at module level) ---
class RetrieveParams(BaseModel):
    kb: str = Field(..., description="Knowledge base to query")
    question: str = Field(..., description="Natural-language query.")
    mode: str = Field("mix", description='Retrieval mode ("mix", "local", "global", "hybrid", "naive", "bypass") default: "mix"')
    top_k: int = Field(30, ge=5, le=120, description="Number of query results to return (5-120).")
    ids: Optional[List[str]] = Field(None, description="Restrict search to these document IDs.")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed_modes = {"mix", "local", "global", "hybrid", "naive", "bypass"}
        if v not in allowed_modes:
            raise ValueError(f"Invalid mode '{v}'. Must be one of {allowed_modes}")
        return v

class AnswerParams(BaseModel):
    kb: str = Field(..., description="Knowledge base to query.")
    question: str = Field(..., description="Natural-language question.")
    mode: str = Field("mix", description='Retrieval mode ("mix", "local", "global", "hybrid", "naive", "bypass") default: "mix".')
    top_k: int = Field(30, ge=5, le=120, description="Number of query results to consider by the answer (5-120).")
    response_type: str = Field("Multiple Paragraphs", description='Answer style ("Multiple Paragraphs", "Single Paragraph", "Bullet Points").')
    ids: Optional[List[str]] = Field(None, description="Limit to specific document IDs.")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed_modes = {"mix", "local", "global", "hybrid", "naive", "bypass"}
        if v not in allowed_modes:
            raise ValueError(f"Invalid mode '{v}'. Must be one of {allowed_modes}")
        return v

    @field_validator('response_type')
    @classmethod
    def validate_response_type(cls, v: str) -> str:
        allowed_types = {"Multiple Paragraphs", "Single Paragraph", "Bullet Points"}
        if v not in allowed_types:
            raise ValueError(f"Invalid response_type '{v}'. Must be one of {allowed_types}")
        return v


# --- Knowledge MCP Service Class ---
class MCP:
    """Encapsulates MCP tools for knowledge base interaction."""
    def __init__(self, rag_manager: RagManager, kb_manager: KnowledgeBaseManager):
        if not isinstance(rag_manager, RagManager):
            raise TypeError("Invalid RagManager instance provided")
        if not isinstance(kb_manager, KnowledgeBaseManager):
            raise TypeError("Invalid KnowledgeBaseManager instance provided")
        self.rag_manager = rag_manager
        self.kb_manager = kb_manager # Store kb_manager if needed for other tools
        self.mcp_server = FastMCP(
            title="Knowledge Base MCP",
            description="Provides tools to search multiple custom knowledge bases.",
            version="0.1.0",
        )
        self.mcp_server.add_tool(
            self.retrieve, 
            name="retrieve", 
            description="Retrieve raw context passages from a knowledge-base without generating an LLM answer."
        )
        self.mcp_server.add_tool(
            self.answer, 
            name="answer", 
            description="Generate an LLM-written answer using the chosen knowledge-base and return it with citations."
        )
        self.mcp_server.add_tool(
            self.list_knowledgebases,
            name="list_knowledgebases",
            description="List all available knowledge bases."
            # No parameters needed, so no model specified
        )
        self.mcp_server.run(transport="stdio")
        logger.info("MCP service initialized.")

    async def retrieve(self, params: RetrieveParams) -> str:
        """
        Retrieve raw context passages from a knowledge‑base without generating an LLM answer.
        """
        logger.info(f"Executing kb_retrieve for KB '{params.kb}' with query: '{params.question[:50]}...' Mode: {params.mode}, TopK: {params.top_k}")
        # Prepare kwargs for rag_manager.query
        query_kwargs = params.model_dump(exclude={'kb', 'question'}, exclude_none=True)
        query_kwargs['only_need_context'] = True
        
        try:
            # Call the now async query method
            context_result: str = await self.rag_manager.query(
                kb_name=params.kb,
                query_text=params.question,
                **query_kwargs
            )
            logger.info(f"Successfully retrieved context for KB '{params.kb}'. Length: {len(context_result)}")
        except (KnowledgeBaseNotFoundError, ConfigurationError) as e:
            logger.warning(f"Configuration or KB not found error during kb_retrieve for '{params.kb}': {e}")
            raise ValueError(str(e)) from e # FastMCP expects ValueError for user input/config issues
        except RAGManagerError as e:
            logger.error(f"Runtime RAG error during kb_retrieve for '{params.kb}': {e}", exc_info=True)
            raise RuntimeError(f"Query failed: {e}") from e # FastMCP expects RuntimeError for internal server errors
        except Exception as e:
            logger.exception(f"Unexpected error during kb_retrieve for '{params.kb}': {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

        return _wrap_result(context_result)

    async def answer(self, params: AnswerParams) -> str:
        """
        Generate an LLM‑written answer using the chosen knowledge‑base and return it with citations.
        """
        logger.info(f"Executing kb_answer for KB '{params.kb}' with query: '{params.question[:50]}...' Mode: {params.mode}, TopK: {params.top_k}, Type: {params.response_type}")
        # Prepare kwargs for rag_manager.query
        query_kwargs = params.model_dump(exclude={'kb', 'question'}, exclude_none=True)
        query_kwargs['only_need_context'] = False

        try:
            # Call the now async query method
            answer: str = await self.rag_manager.query(
                kb_name=params.kb,
                query_text=params.question,
                **query_kwargs
            )
            logger.info(f"Successfully generated answer for KB '{params.kb}'. Length: {len(answer)}")
        except (KnowledgeBaseNotFoundError, ConfigurationError) as e:
            logger.warning(f"Configuration or KB not found error during kb_answer for '{params.kb}': {e}")
            raise ValueError(str(e)) from e
        except RAGManagerError as e:
            logger.error(f"Runtime RAG error during kb_answer for '{params.kb}': {e}", exc_info=True)
            raise RuntimeError(f"Query failed: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during kb_answer for '{params.kb}': {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

        return _wrap_result(answer)

    async def list_knowledgebases(self) -> str:
        """Lists all available knowledge bases and their descriptions."""
        logger.info("Executing list_knowledgebases")
        try:
            # kb_manager.list_kbs is now async and returns Dict[str, str]
            kb_dict: Dict[str, str] = await self.kb_manager.list_kbs()
            logger.info(f"Found knowledge bases: {kb_dict}")

            # Transform the dict into the desired list of objects format
            kb_list_formatted = [
                {"name": name, "description": description}
                for name, description in kb_dict.items()
            ]

            # Wrap in the final structure and return as JSON
            result = {"knowledge_bases": kb_list_formatted}
            return json.dumps(result)

        except KnowledgeBaseError as e:
            logger.error(f"Error listing knowledge bases: {e}", exc_info=True)
            # Use ValueError for user-facing errors expected by FastMCP
            raise ValueError(f"Failed to list knowledge bases: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during list_knowledgebases: {e}")
            # Use RuntimeError for internal server errors expected by FastMCP
            raise RuntimeError(f"An unexpected server error occurred: {e}") from e
