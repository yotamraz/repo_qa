"""
The purpose of this script is to demonstrate the power of agentic LLMs.
Coupling python function the LLM can choose to execute to achieve a certain goals.

This script is an example of how repo_qa tool may be used for code review purposes
"""

import os
import argparse
import multiprocessing

import openai
import requests
from loguru import logger
from dotenv import load_dotenv

from repo_qa.utils import run_api_server, wait_for_server, get_git_diff

# Define a "function" that the LLM can call to query your QA server
# We'll provide a schema with a name, description, and JSON parameters.
functions = [
    {
        "name": "ask_code_qa_server",
        "description": "Send a question to the RAG-based code QA server and get the answer back.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A question about the codebase"
                }
            },
            "required": ["question"]
        }
    }
]


def ask_code_qa_server(question: str, qa_url="http://0.0.0.0:8000/query_repo") -> str:
    """
    Calls your existing QA server's /answer endpoint with the given question,
    returns the answer text.
    """
    try:
        resp = requests.post(qa_url, json={"question": question})
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as e:
        return f"[ERROR calling QA server: {e}]"


def generate_code_review(max_iterations: int, diff_text: str) -> str:
    """
    This is the main function the user calls. We'll do a conversation with an LLM that can:
    - See the diff
    - Potentially call 'ask_code_qa_server' function multiple times for context
    - Return a final "code review" style commentary
    # TODO: a nice feature will be adding a tool to post comments on github so the human developer
            may start working on remarks.
            didn't have enough time for that...
    """

    system_prompt = (
        "You are a senior software engineer doing a code review (pull request before merge) for other engineers on the team. You can call a function "
        "to ask the RAG-based code QA server for context about the code base. "
        "Use any discovered info (calling the function as many times as needed) to produce "
        "a thorough code review of the changes in the user's diff."
        "The review should be constructive, and should include fix suggestions if possible (as code snippets or natural language comments, preferably both)"
    )

    user_prompt = (
        f"The user provides the following diff. Please review it for correctness, style, "
        f"and any potential issues:\n\n{diff_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # We'll do a loop to let the model function-call the QA server as needed
    # and eventually produce a final answer.
    #
    # We'll break out of the loop if the model doesn't request the function again.
    #
    # This is a simple approach. You can implement more advanced "agent" loops if needed.
    for _ in range(max_iterations):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",  # Let the LLM decide when/if to call
            temperature=0.2
        )

        # If the LLM didn't call the function, we should have our final answer
        if "choices" in response and len(response["choices"]) > 0:
            finish_reason = response["choices"][0]["finish_reason"]
            msg = response["choices"][0]["message"]

            if finish_reason == "stop" or finish_reason is None:
                # The LLM is providing a final answer
                return msg["content"]

            if msg.get("function_call"):
                # The LLM wants to call a function
                fn_name = msg["function_call"]["name"]
                if fn_name == "ask_code_qa_server":
                    # The LLM is requesting context from the QA server
                    fn_args = eval(msg["function_call"]["arguments"])
                    question = fn_args.get("question", "")
                    # call the actual function
                    answer = ask_code_qa_server(question)
                    # feed the result back into the conversation
                    messages.append(msg)  # the function call request
                    messages.append({
                        "role": "function",
                        "name": fn_name,
                        "content": answer
                    })
                else:
                    # Unexpected function name
                    messages.append({
                        "role": "assistant",
                        "content": f"Error: function '{fn_name}' not recognized."
                    })
            else:
                # No function call, so let's assume final
                return msg.get("content", "")
        else:
            # fallback
            return "Error: No response from LLM."

    return "Error: Reached max iterations without a final answer."

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str, required=True, help="Path to a local git repo")
    parser.add_argument("--env_file", type=str, default="./.env", help="Path to .env file")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum number of agent iterations")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server address")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    return parser.parse_args()

def main():
    args = arg_parse()
    load_dotenv(args.env_file)
    openai.api_key = os.getenv("OPENAI_API_KEY", None)
    proc = multiprocessing.Process(target=run_api_server, args=(args.host, args.port,))
    proc.start()

    url = f"http://{args.host}:{args.port}"

    wait_for_server(f"{url}/health")

    # index given repo
    requests.post(url=f"{url}/index_repo", json={"repo_path": args.repo_path})

    # query the system with reference Q&A
    ## RUN AGENT
    git_diff = get_git_diff(args.repo_path)
    results = generate_code_review(max_iterations=args.max_iterations, diff_text=git_diff)
    logger.info(results)

    # stop API server
    proc.terminate()


if __name__ == '__main__':
    main()

