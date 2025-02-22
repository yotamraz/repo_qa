# repo_qa

## Overview

`repo_qa` is a tool designed to facilitate Q&A on any Python repository using Large Language Models (LLMs). It leverages modern AI capabilities to provide insights and answers about the codebase.

## Features

- **FastAPI**: Utilizes FastAPI to create a web server for handling requests.
- **OpenAI Integration**: Connects with OpenAI's API to utilize language models for processing queries.
- **ChromaDB**: Uses ChromaDB for efficient data storage and retrieval.
- **Logging**: Implements `loguru` for enhanced logging capabilities.
- **Environment Management**: Supports `.env` files for managing environment variables securely.

## Installation

Ensure you have Python 3.12 or higher installed. You can install `repo_qa` using Poetry:

```bash
pip install poetry
cd repo_qa
poetry install
```

## Usage

To start the `repo_qa` server, run:

```bash
repo_qa --env_file PATH_TO_ENV_FILE --host 0.0.0.0 --port 8000
```

This will launch the FastAPI server, allowing you to interact with the tool via HTTP requests.

## Configuration
Create a `.env` file in the root directory with the following variables:

```plaintext
OPENAI_API_KEY=YOUR-KEY-GOES-HERE
EMBEDDING_MODEL_NAME=text-embedding-ada-002
CHAT_MODEL_NAME=gpt-4o
```
Replace `YOUR-KEY-GOES-HERE` with your actual OpenAI API key.

## Dependencies
The project relies on several key dependencies:
- `fastapi`:^0.85.1
- `uvicorn`: ^0.34.0
- `openai`: ^0.28
- `chromadb`: ^0.3.29
- `requests`: ^2.32.3
- `python-dotenv`: ^1.0.1
- `loguru`: ^0.7.3
- `tqdm`: ^4.67.1

## License
This project is licensed under the MIT License

## Author
Developed by Yotam Raz. Contact:yotamraz2007@gmail.com