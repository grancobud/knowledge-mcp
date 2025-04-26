from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
import os

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Use model_name from global config
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        keyword_extraction=keyword_extraction,
        **kwargs,
    )

async def openai_embedding_func(texts: list[str]) -> np.ndarray:
    embedding_model = kwargs.get("embedding_model", embedding_config["model_name"])
    
    # Call the OpenAI embed function with all the parameters
    return await openai_embed(
        texts=texts,
        model=embedding_model,
        api_key=os.getenv("OPENAI_API_KEY"),  # Or from config
        base_url=os.getenv("OPENAI_API_BASE"),  # Or from config
    )

# Wrap the embedding function with the correct attributes from config
embedding_func = EmbeddingFunc(
    embedding_dim=embedding_config.get("embedding_dim", 1536),
    max_token_size=embedding_config.get("max_token_size", 8192)
)(openai_embedding_func)