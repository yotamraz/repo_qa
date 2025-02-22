"""
To run the evaluation script repo_qa package must be installed
"""

import time
import multiprocessing
import argparse
import json

import requests
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from repo_qa.api import app

def wait_for_server(url, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("API server is up!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    logger.info("API server did not start in time.")
    return False

def evaluate(endpoint="http://0.0.0.0:8000/query_repo", reference_file="reference_qa.json"):
    """
    1. Reads question/answer pairs from reference_file.
    2. For each question, calls our service at api_url.
    3. Compares system answer to reference answer using ROUGE-L.
    4. Prints average ROUGE-L F1 over all pairs.
    """

    # Load reference QAs
    with open(reference_file, "r") as f:
        references = json.load(f)

    # Initialize ROUGE scorer for ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_1_f_score_total = 0.0
    rouge_2_f_score_total = 0.0
    rouge_l_f_score_total = 0.0
    count = 0

    for idx, item in enumerate(references):
        question = item["question"]
        reference_answer = item["reference_answer"]

        # 1. Call your QA service
        try:
            response = requests.post(endpoint, json={"question": question})
            response.raise_for_status()
            system_answer = response.json().get("answer", "")
        except Exception as e:
            logger.error(f"[Error] Could not get answer for Q{idx} -> {e}")
            continue

        # compute ROUGE-L
        # - We compare reference answer vs. system answer
        scores = scorer.score(reference_answer, system_answer)
        rouge_1_f_score = scores["rouge1"].fmeasure
        rouge_2_f_score = scores["rouge2"].fmeasure
        rouge_l_f_score = scores["rougeL"].fmeasure

        rouge_1_f_score_total += rouge_1_f_score
        rouge_2_f_score_total += rouge_2_f_score
        rouge_l_f_score_total += rouge_l_f_score
        count += 1

        logger.info(f"Q{idx} question:\n {question}")
        logger.info(f"Q{idx} reference answer:\n {reference_answer}")
        logger.info(f"Q{idx} generated answer:\n {system_answer}")
        logger.info(f"Q{idx} -> ROUGE-1 F1: {rouge_1_f_score:.3f}")
        logger.info(f"Q{idx} -> ROUGE-2 F1: {rouge_2_f_score:.3f}")
        logger.info(f"Q{idx} -> ROUGE-L F1: {rouge_l_f_score:.3f}")

    if count > 0:
        rouge_1_f_score_avg = rouge_1_f_score_total / count
        rouge_2_f_score_avg = rouge_2_f_score_total / count
        rouge_l_f_score_avg = rouge_l_f_score_total / count
        logger.info(f"Average ROUGE-1 F1 across {count} questions: {rouge_1_f_score_avg:.3f}")
        logger.info(f"Average ROUGE-2 F1 across {count} questions: {rouge_2_f_score_avg:.3f}")
        logger.info(f"Average ROUGE-L F1 across {count} questions: {rouge_l_f_score_avg:.3f}")
    else:
        logger.info("No valid reference questions evaluated.")

def run_api_server(host: str, port: int):
    uvicorn.run(app, host=host, port=port)

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str, required=True, help="Path to a local git repo")
    parser.add_argument("--env_file", type=str, default="./.env", help="Path to .env file")
    parser.add_argument("--reference_file_path", type=str, default="reference_qa.json", help="Path to reference qa json file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server address")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    return parser.parse_args()

def main():
    args = arg_parse()
    load_dotenv(args.env_file)
    proc = multiprocessing.Process(target=run_api_server, args=(args.host, args.port,))
    proc.start()

    url = f"http://{args.host}:{args.port}"

    wait_for_server(f"{url}/health")

    # index given repo
    requests.post(url=f"{url}/index_repo", json={"repo_path": args.repo_path})

    # query the system with reference Q&A
    evaluate(endpoint=f"{url}/query_repo", reference_file=args.reference_file_path)

    # stop API server
    proc.terminate()


if __name__ == '__main__':
    main()