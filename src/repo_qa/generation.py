

import openai

from .config import SystemConfig

def generate_answer(question: str, retrieved_chunks: list, openai_api_key: str, chat_model_name: str):
    """
    'retrieved_chunks' is a list of (chunk_id, chunk_text, metadata).
    We'll build a prompt with the chunk texts, plus the question.
    """
    openai.api_key = openai_api_key
    # build context string
    context_pieces = []
    for chunk_id, chunk_text, meta in retrieved_chunks:
        file_path = meta.get("file_path", "")
        name = meta.get("name", "")
        context_pieces.append(
            f"\n\n--- Chunk from {file_path} ({name}):\n{chunk_text}\n"
        )

    context_string = "".join(context_pieces)

    system_prompt = (
        "You are a helpful assistant that answers questions about Python code. "
        "You have access to the following code snippets (and their relationships). "
        "Use these to provide a concise, correct answer. Include code snippets if needed."
    )
    user_prompt = f"QUESTION:\n{question}\nCODE CONTEXT:\n{context_string}\n\nANSWER:"

    response = openai.ChatCompletion.create(
        model=chat_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=SystemConfig.max_generation_tokens,
        temperature=SystemConfig.generation_temperature
    )
    return response["choices"][0]["message"]["content"]
