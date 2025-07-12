## Relevant Files

- `knowledge_mcp/knowledgebases.py` - Contains DEFAULT_QUERY_PARAMS and config loading logic that needs to be updated
- `knowledge_mcp/rag.py` - Contains the query method that needs to handle user_prompt parameter
- `kbs/*/config.yaml` - Knowledge base configuration files that will include the new user_prompt field

### Notes

- No unit tests are required for this feature as per user preferences
- Changes are primarily configuration and parameter passing modifications
- Existing knowledge bases will continue to work without modification due to backward compatibility design

## Tasks

- [x] 1.0 Update Default Query Parameters Configuration
  - [x] 1.1 Add `user_prompt: ""` field to DEFAULT_QUERY_PARAMS dictionary
  - [x] 1.2 Change default `mode` from "mix" to "hybrid"
  - [x] 1.3 Change default `top_k` from 60 to 40
- [ ] 2.0 Implement User Prompt Integration in Query Processing
  - [ ] 2.1 Modify query method to extract user_prompt from kb_config
  - [ ] 2.2 Add user_prompt to final_query_params when not empty
  - [ ] 2.3 Ensure QueryParam instance includes user_prompt parameter
- [ ] 3.0 Add Configuration Loading Support for User Prompts
  - [ ] 3.1 Update config loading logic to include user_prompt field
  - [ ] 3.2 Ensure user_prompt is filtered and passed through properly
  - [ ] 3.3 Handle missing user_prompt field gracefully (backward compatibility)
- [ ] 4.0 Add Logging and Debug Support for User Prompt Application
  - [ ] 4.1 Add debug logging when user_prompt is loaded from config
  - [ ] 4.2 Add debug logging when user_prompt is applied to QueryParam
  - [ ] 4.3 Use knowledge-base-specific loggers for user_prompt messages
- [ ] 5.0 Test and Validate Implementation
  - [ ] 5.1 Test with empty user_prompt (backward compatibility)
  - [ ] 5.2 Test with configured user_prompt (functionality)
  - [ ] 5.3 Verify logging output and default parameter changes
