from typing import Optional

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings
from loguru import logger
from tqdm import tqdm

from .chunking import extract_code_blocks
from .callgraph import build_call_graph

class CoherentChunkOpenAIEmbeddingFunction(embedding_functions.OpenAIEmbeddingFunction):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-ada-002", organization_id: Optional[str] = None, api_base: Optional[str] = None, api_type: Optional[str] = None, max_chunk_size: int = 8192):
        super().__init__(api_key, model_name, organization_id, api_base, api_type)
        self.max_chunk_size = max_chunk_size

    def __call__(self, texts: Documents) -> Embeddings:
        # Replace newlines to avoid negative performance impact
        texts = [t.replace("\n", " ") for t in texts]

        # Define the maximum allowed chunk size.
        # Adjust this value based on your embedding model’s token/character limit.

        # Prepare a list for all text chunks and a mapping of each chunk to its document index.
        chunk_texts = []
        doc_chunk_map = []
        for doc_index, text in enumerate(texts):
            # If the document is longer than the allowed chunk size, split it;
            # otherwise, use the whole text.
            if len(text) > self.max_chunk_size:
                chunks = [text[i:i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
            else:
                chunks = [text]
            chunk_texts.extend(chunks)
            doc_chunk_map.extend([doc_index] * len(chunks))

        # Call the embedding API on the list of chunks.
        response = self._client.create(input=chunk_texts, engine=self._model_name)
        embeddings_data = response["data"]

        # Sort the embeddings by the provided index to ensure order consistency.
        sorted_embeddings = sorted(embeddings_data, key=lambda e: e["index"])

        # Group chunk embeddings by the original document.
        doc_embeddings = {}
        for embedding_info, doc_idx in zip(sorted_embeddings, doc_chunk_map):
            vector = np.array(embedding_info["embedding"])
            doc_embeddings.setdefault(doc_idx, []).append(vector)

        # Merge embeddings for each document by averaging.
        final_embeddings = []
        for i in range(len(texts)):
            vectors = doc_embeddings.get(i, [])
            if vectors:
                merged_vector = np.mean(vectors, axis=0)
                final_embeddings.append(merged_vector.tolist())
            else:
                # In case no embeddings were computed for a document, append an empty list.
                final_embeddings.append([])

        return final_embeddings



def build_index(repo_path: str,
                db_dir: str = "./db_dir",
                collection_name: str = "code_chunks",
                openai_api_key: str = "None",
                embedding_model_name: str = "text-embedding-ada-002"
                ):
    """
    1) Build call graph
    2) Extract code blocks
    3) Embed code blocks into a vector DB
    4) Return (collection, call_graph)
    """
    # 1. Build call graph
    logger.info(f"building a call-graph for {repo_path}")
    call_graph, defined_funcs = build_call_graph(repo_path)

    # 2. Initialize Chroma
    logger.info(f"starting a ChromaDB, persist dir: {db_dir}")
    client = chromadb.Client(
        settings=chromadb.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_dir
        )
    )

    # 3. Embedding function
    logger.info(f"creating an embedding function using {embedding_model_name} model")
    embedder = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model_name,
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder
    )

    # 4. Extract code blocks & embed them
    logger.info("storing code chunks in db with call-graph info")
    i = 0
    for code_block, metadata in tqdm(extract_code_blocks(repo_path)):
        doc_id = f"chunk_{i}"
        collection.add(
            documents=[code_block],
            metadatas=[metadata],
            ids=[doc_id]
        )
        i += 1

    print("Index build complete. Collection size =", collection.count())
    return collection, call_graph
