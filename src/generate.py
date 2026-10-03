"""Gemini generation wrapper."""

import os

from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Ask Gemini to answer using only the retrieved chunks."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    context = "\n\n".join(
        f"Source: {chunk['source']}, page {chunk['page']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    prompt = f"""Answer the question using only the context below.
If the context does not contain enough information, say that you do not know.
Do not invent information. Include relevant source filenames and page numbers
in your answer.

Question:
{question}

Context:
{context}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        contents=prompt,
    )
    return response.text
