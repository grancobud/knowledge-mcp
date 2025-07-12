# PRD: Configurable User Prompts for Knowledge Bases

## Introduction/Overview

This feature adds the ability for users to configure custom user prompts per knowledge base that will be automatically applied to all queries. This leverages LightRAG's built-in user prompt functionality (using square bracket syntax) to allow users to control how the LLM synthesizes answers from their knowledge base without affecting the retrieval phase.

The feature addresses the need for consistent, knowledge-base-specific prompt engineering that can guide the LLM's output formatting, tone, or content processing approach for different domains or use cases.

## Goals

1. **Per-KB Customization**: Enable users to set custom user prompts for each knowledge base independently
2. **Seamless Integration**: Automatically apply configured user prompts to all queries without manual intervention
3. **Backward Compatibility**: Ensure existing knowledge bases continue to work without modification
4. **Configuration Simplicity**: Use the existing config.yaml structure for easy configuration
5. **Default Updates**: Update default query parameters to more optimal values (mode: hybrid, top_k: 30)

## User Stories

1. **As a technical documentation maintainer**, I want to configure my knowledge base to always format responses using markdown and include code examples, so that answers are consistently well-structured for developers.

2. **As a customer support manager**, I want my knowledge base to always respond in a friendly, helpful tone with step-by-step instructions, so that customer queries get consistent, professional responses.

3. **As a research analyst**, I want my knowledge base to always provide detailed citations and confidence levels in responses, so that I can trust and verify the information provided.

4. **As a content creator**, I want my knowledge base to format responses as bullet points with actionable insights, so that the information is easily digestible for my audience.

5. **As a system administrator**, I want to update my existing knowledge bases with better default settings without breaking existing functionality, so that query performance improves automatically.

## Functional Requirements

1. **Configuration Field Addition**
   - Add `user_prompt` field to the DEFAULT_QUERY_PARAMS dictionary
   - Set default value to empty string (`""`)
   - Support multi-line prompt content in YAML format

2. **Query Integration**
   - Automatically prepend user_prompt to queries using LightRAG's square bracket syntax: `[user_prompt] actual_query`
   - Only apply user_prompt when it exists and is not empty (avoid altering queries when user_prompt is empty string)
   - Always use the user_prompt from the knowledge base's config.yaml file

3. **Default Parameter Updates**
   - Change default `mode` from "mix" to "hybrid"
   - Change default `top_k` from 60 to 40
   - Keep all other default parameters unchanged

4. **Configuration Loading**
   - Include `user_prompt` in the config loading and filtering logic
   - Ensure `user_prompt` is passed through to query parameters
   - Handle missing `user_prompt` field gracefully (treat as empty string)

5. **Logging Enhancement**
   - Add debug logging to show when user_prompt is being applied to queries
   - Log the actual prompt being used for debugging purposes
   - Use knowledge-base-specific loggers for prompt application messages

6. **No Validation Requirements**
   - No length limits or format validation on user_prompt content
   - Allow any string content including special characters and formatting

## Non-Goals (Out of Scope)

1. **Runtime Prompt Override**: Users cannot override the configured user_prompt at query time
2. **Global User Prompts**: No system-wide default user prompts across all knowledge bases
3. **Prompt Templates**: No templating system or variable substitution in user prompts
4. **Migration Tools**: No automatic migration of existing config files
5. **UI Configuration**: No web interface for setting user prompts (config.yaml only)
6. **Prompt History**: No tracking or versioning of user prompt changes

## Technical Considerations

1. **LightRAG Integration**: Use LightRAG's existing square bracket syntax for user prompts
2. **Query Modification**: Modify the query string before passing to LightRAG's query method
3. **Configuration Structure**: Extend existing config.yaml structure without breaking changes
4. **Thread Safety**: Ensure user_prompt application works correctly in async/threaded query execution
5. **Memory Efficiency**: No additional caching needed for user prompts (loaded with config)

## Success Metrics

1. **Functionality**: 100% of queries with configured user_prompts are properly formatted
2. **Backward Compatibility**: 100% of existing knowledge bases continue to work without modification
3. **Performance**: No measurable impact on query response times
4. **Adoption**: User prompts can be successfully configured and applied across different knowledge base types
5. **Logging**: Debug logs clearly show when and how user prompts are applied

## Implementation Details

### Files to Modify

1. **`knowledge_mcp/knowledgebases.py`**:
   - Update `DEFAULT_QUERY_PARAMS` to include `user_prompt: ""`
   - Update default `mode` to "hybrid" and `top_k` to 40
   - Ensure `user_prompt` is included in config loading logic

2. **`knowledge_mcp/rag.py`**:
   - Modify the `query` method to include user_prompt in QueryParam when not empty
   - Add logging for user_prompt application
   - Handle empty user_prompt gracefully (don't include in QueryParam if empty)
   - No modification to query_text needed - user_prompt is passed via QueryParam

### Query Processing Logic

```python
# In the query method, add user_prompt to QueryParam if not empty
if user_prompt and user_prompt.strip():
    final_query_params['user_prompt'] = user_prompt

# QueryParam instance creation will include user_prompt
query_param_instance = QueryParam(**final_query_params)

# Query execution remains the same - no modification to query_text needed
result = await asyncio.to_thread(
    fresh_rag_instance.lightrag.query,
    query_text,  # Original query text unchanged
    query_param_instance  # QueryParam now includes user_prompt
)
```

## Open Questions

1. Should there be any escaping or sanitization of square brackets within user_prompt content to avoid conflicts with LightRAG syntax?
2. Should the logging level for user_prompt application be configurable, or is DEBUG level appropriate for all cases?
3. Should there be any documentation or examples provided for effective user prompt patterns?

---

**Status**: Ready for Implementation  
**Priority**: Medium  
**Estimated Effort**: 2-3 hours  
**Dependencies**: None
