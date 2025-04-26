from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
import os
from knowledge_mcp.config import Config

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Use model_name from global config provided by LightRAG within kwargs
    # Check if 'hashing_kv' and 'global_config' are present before accessing
    model_name = kwargs["hashing_kv"].global_config.get("llm_model_name")

    # Prioritize api_key and base_url from kwargs (coming from llm_model_kwargs)
    # Use environment variables as fallback
    # api_key = kwargs.pop('api_key', os.getenv("OPENAI_API_KEY"))
    # base_url = kwargs.pop('base_url', os.getenv("OPENAI_API_BASE"))

    api_key = Config.get_instance().lightrag.llm.api_key
    base_url = Config.get_instance().lightrag.llm.api_base

    if not api_key:
        raise ValueError("OpenAI API key is not provided in config or environment variables.")

    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key, # Pass the determined api_key
        base_url=base_url, # Pass the determined base_url (can be None)
        keyword_extraction=keyword_extraction,
        **kwargs, # Pass remaining kwargs
    )

async def openai_embedding_func(texts: list[str]) -> np.ndarray:
    embedding_model = Config.get_instance().lightrag.embedding.model_name
    
    # Call the OpenAI embed function with all the parameters
    return await openai_embed(
        texts=texts,
        model=embedding_model,
        api_key=Config.get_instance().lightrag.embedding.api_key,
        base_url=Config.get_instance().lightrag.embedding.api_base,
    )

# Wrap the embedding function with the correct attributes from config
embedding_func = EmbeddingFunc(
    embedding_dim=Config.get_instance().lightrag.embedding.embedding_dim,
    max_token_size=Config.get_instance().lightrag.embedding.max_token_size,
    func=openai_embedding_func
)