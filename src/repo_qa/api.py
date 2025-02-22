import argparse
import os

import uvicorn
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from loguru import logger

from .indexing import build_index
from .retrieval import retrieve_with_callgraph
from .generation import generate_answer

app = FastAPI()

# Load the .env file from the base repo level.

# Global variables to store the collection + call graph
collection = None
call_graph = None

@app.post("/index_repo")
def index_repo(payload: dict = Body(...)):
    """
    Build the index and call graph when the app starts.
    In a production environment, you might do this differently or
    cache/persist the DB so you don't re-build every time.
    """

    global collection, call_graph
    repo_path = payload["repo_path"]
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"repo path does not exist: {repo_path}")
    logger.info(f"starting indexing repo: {repo_path}")
    col, cg = build_index(
        repo_path=repo_path,
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", None)
    )
    collection = col
    call_graph = cg

    logger.info(f"finished indexing repo: {repo_path}")
    return JSONResponse(content={"message": f"Successfully indexed {repo_path}"})

@app.post("/query_repo")
def query_repo(payload: dict = Body(...)):
    """
    Expects {"question": "..."}
    Returns: {"answer": "..."}
    """
    question = payload["question"]
    logger.info(f"building answer for user question: {question}")

    # 1. Retrieve with callgraph
    retrieved = retrieve_with_callgraph(question, collection, call_graph)
    # 2. Generate answer
    logger.info(f"generating final answer")
    final_answer = generate_answer(
        question,
        retrieved,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model_name=os.getenv("CHAT_MODEL_NAME")
    )
    return JSONResponse(content={"answer": final_answer})

@app.get("/health")
def health():
    return JSONResponse(content={})

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_file", type=str, default="./.env", help="path to .env file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="api host")
    parser.add_argument("--port", type=int, default=8000, help="api port")

    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.env_file):
        logger.error(f".env file not found: {args.env_file}")
        raise ValueError(f".env file not found: {args.env_file}")

    load_dotenv(dotenv_path=args.env_file)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
