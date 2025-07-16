# Migration Guide: lightrag-hku 1.4.0 Breaking Changes

This guide helps you update your knowledge base configurations for lightrag-hku 1.4.0 compatibility.

## What Changed

lightrag-hku 1.4.0 introduced breaking changes to the `QueryParam` class:

### Parameter Name Changes
- `max_token_for_text_unit` → `max_entity_tokens`
- `max_token_for_global_context` → `max_relation_tokens` 
- `max_token_for_local_context` → `max_total_tokens`

### Default Mode
- knowledge-mcp continues to use `"hybrid"` as the default mode (though `"mix"` is available)

### New Parameters Added
- `stream`: Enable streaming output (default: `false`)
- `chunk_top_k`: Number of text chunks to retrieve (default: `null`, uses `top_k`)
- `hl_keywords`: High-level keywords list (default: `[]`)
- `ll_keywords`: Low-level keywords list (default: `[]`)
- `conversation_history`: Conversation history (default: `[]`)
- `ids`: List of IDs to filter results (default: `null`)
- `model_func`: Optional LLM model function override (default: `null`)
- `enable_rerank`: Enable reranking for text chunks (default: `true`)

## How to Update Your Config Files

### Before (1.3.9)
```yaml
description: "My research papers"
mode: "hybrid"
top_k: 30
max_token_for_text_unit: 2000
max_token_for_global_context: 3000
max_token_for_local_context: 4000
response_type: "Multiple Paragraphs"
user_prompt: "Please provide detailed analysis."
```

### After (1.4.0)
```yaml
description: "My research papers"
mode: "hybrid"  # knowledge-mcp default ("mix" is also available)
top_k: 30
max_entity_tokens: 2000
max_relation_tokens: 3000
max_total_tokens: 6000  # Consider increasing for better context
response_type: "Multiple Paragraphs"
user_prompt: "Please provide detailed analysis."
# Optional new parameters:
# stream: false
# enable_rerank: true
# hl_keywords: []
# ll_keywords: []
```

## Automatic Migration

The knowledge-mcp system now includes automatic migration functionality:

1. **Automatic Migration**: When you load any knowledge base, the system automatically checks for old parameter names and migrates them to the new format.
2. **Backup Creation**: Before migration, a backup of your original config is created as `config.yaml.backup`.
3. **New Knowledge Bases**: When you create new knowledge bases, they automatically use the 1.4.0-compatible configuration.

## Existing Knowledge Bases

**Good news**: Migration happens automatically! When you use any knowledge base, the system will:

1. **Detect old parameters** in your `config.yaml` files
2. **Create a backup** (`config.yaml.backup`) of your original config
3. **Update the config** with new parameter names
4. **Log the migration** so you know what happened

If you want to migrate all your knowledge bases at once, you can also do it manually:

```python
from knowledge_mcp.knowledgebases import KnowledgeBaseManager
from knowledge_mcp.config import Config

config = Config.from_file("config.yaml")
kb_manager = KnowledgeBaseManager(config)
results = kb_manager.migrate_all_configs()
print(f"Migration results: {results}")
```

## Mode Comparison

| Mode | Description | Best For |
|------|-------------|----------|
| `hybrid` | **knowledge-mcp default** - Combines local and global retrieval methods | Cross-domain queries |
| `mix` | Integrates knowledge graph and vector retrieval | General purpose queries |
| `local` | Focuses on context-dependent information | Specific, targeted queries |
| `global` | Utilizes global knowledge | Broad, conceptual queries |

## Need Help?

If you encounter issues after upgrading:

1. Check that your `config.yaml` files use the new parameter names
2. Verify that `mode` is set to a valid value (`"mix"`, `"hybrid"`, `"local"`, `"global"`, `"naive"`, or `"bypass"`)
3. Ensure token limits are reasonable (try the defaults first)

## Rollback

If you need to rollback to 1.3.9:
```bash
pip install lightrag-hku==1.3.9
```

Then revert your config files to use the old parameter names.
