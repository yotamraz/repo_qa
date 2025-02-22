
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
    top_k_entities = 3
    max_callgraph_depth = 1

    # chunking
    file_suffixes = [".py", ".ipynb", ".toml", ".ini", ".md", ".yml", "yaml", ""]
