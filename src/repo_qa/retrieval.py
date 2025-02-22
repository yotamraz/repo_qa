
import chromadb
from loguru import logger

from .indexing import build_index
from .config import SystemConfig

def retrieve_with_callgraph(question: str,
                            collection,
                            call_graph):
    """
    1. Vector-based retrieval of top-k chunks
    2. For each chunk, collect call-graph neighbors up to 'expansion_depth'
    3. Merge & re-rank or limit them
    """
    logger.info(f"fetching top {SystemConfig.top_k_entities} results from index")
    # 1. Vector-based retrieval
    results = collection.query(query_texts=[question], n_results=SystemConfig.top_k_entities)

    top_docs = results["documents"][0]
    top_metas = results["metadatas"][0]
    top_ids   = results["ids"][0]

    logger.info(f"gathering neighboring entities from call graph with max depth of {SystemConfig.max_callgraph_depth}")
    # 2. Gather neighbors from the call graph
    # We'll store (chunk_text, meta) in a list. Then optionally re-rank.
    all_candidates = []  # store chunk IDs
    expansions = []

    for doc_id, doc_text, meta in zip(top_ids, top_docs, top_metas):
        all_candidates.append((doc_id, doc_text, meta))
        name = meta.get("name")
        if not name:
            continue
        # BFS or DFS to get neighbors up to 'expansion_depth'
        neighbors = get_graph_neighbors(name, call_graph, SystemConfig.max_callgraph_depth)
        expansions.extend(neighbors)

    # expansions might just be function names. We need to find chunk IDs that match.
    # We'll do a quick filter in the collection by metadata name.
    all_expanded_docs = {"ids": [], "documents": [], "metadatas": []}
    for name in list(expansions):
        result = collection.get(where={"name": {"$eq": name}})
        all_expanded_docs["ids"].extend(result.get("ids", []))
        all_expanded_docs["documents"].extend(result.get("documents", []))
        all_expanded_docs["metadatas"].extend(result.get("metadatas", []))

    # This returns all matching docs. Add them to all_candidates
    for doc_id, doc_text, meta in zip(all_expanded_docs["ids"],
                                      all_expanded_docs["documents"],
                                      all_expanded_docs["metadatas"]):
        all_candidates.append((doc_id, doc_text, meta))

    # TODO: improvement suggestion: run embedding and compare to question again
    return list(all_candidates)

def get_graph_neighbors(start_name, call_graph, depth=1):
    """
    Return a set of node names reachable from 'start_name'
    within 'depth' steps. (Simple BFS approach.)
    """
    visited = set()
    queue = [(start_name, 0)]
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if node in call_graph and dist < depth:
                for neigh in call_graph[node]:
                    queue.append((neigh, dist + 1))
    return visited
