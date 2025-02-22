
class SystemConfig:
    """
    global configuration
    """
    # generation
    max_generation_tokens = 800
    generation_temperature = 0.3

    # indexing
    embedding_function = "OpenAIEmbeddingFunction"

    # retrieval
    top_k_entities = 10
    max_callgraph_depth = 2

    # chunking
    file_suffixes = [".py", ".ipynb", ".toml", ".ini", ".md", ".yml", "yaml", ""]
    max_chunk_size = 8000
