"""Interactive command-line interface for the RAG assistant."""

from generate import generate_answer
from retrieve import retrieve


def main() -> None:
    print("RAG Knowledge Assistant (type 'quit' to exit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        try:
            chunks = retrieve(question)
            answer = generate_answer(question, chunks)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"Error: {error}")
            continue
        print(f"\nAnswer:\n{answer}")
        print("\nRetrieved sources:")
        for chunk in chunks:
            print(f"- {chunk['source']}, page {chunk['page']}")


if __name__ == "__main__":
    main()
