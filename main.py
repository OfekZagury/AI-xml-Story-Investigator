import os
import sys
from dotenv import load_dotenv
from pathlib import Path

from utils.data_loader import load_xml_data
from rag_implementations.naive_rag import NaiveRAG

load_dotenv()
current_dir = Path(__file__).parent


def print_separator():
    print("-" * 50)


def main():
    print_separator()
    print("Welcome to the Multi-Strategy RAG System")
    print_separator()

    # 1. Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file or export the key.")
        sys.exit(1)

    # 2. Load Data
    # Default for quick testing if file exists
    xml_path = current_dir / "whodunit_sydney.xml"
    print(f"Using: {xml_path}")

    try:
        messages = load_xml_data(str(xml_path))
        print(f"Successfully loaded {len(messages)} messages.")
    except Exception as e:
        print(f"Error loading XML: {e}")
        sys.exit(1)

    # 3. Choose Strategy
    print_separator()
    print("Choose your RAG Strategy:")
    print("1. Naive RAG (In-Memory, Simple, Fast)")
    print("2. Microsoft GraphRAG (Batch Processing, Global Summaries)")
    print("3. Neo4j GraphRAG (Database-backed, Graph+Vector)")

    choice = input("Enter number (1-3): ").strip()
    rag_engine = None

    # --- Strategy Initialization ---

    if choice == '1':
        print("\nInitializing Naive RAG...")
        rag_engine = NaiveRAG()
        print("Ingesting data into memory...")
        for mes in messages:
            # Combining title and content for better context
            full_text = f"Title: {mes.get('title', '')}\nContent: {mes.get('content', '')}"
            rag_engine.add_message(full_text)

    elif choice == '2':
        print("\nnot implemented: Microsoft GraphRAG")


    elif choice == '3':
        print("\nnot implemented: Neo4j GraphRAG...")


    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # 4. Q&A Loop
    print_separator()
    print("System Ready. Type 'exit' to quit.")

    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() in ['exit', 'quit']:
            break

        if not question:
            continue

        print("Thinking...")
        try:
            answer = rag_engine.answer_question(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error retrieving answer: {e}")

    # Cleanup (if needed)
    if choice == '3' and hasattr(rag_engine, 'close'):
        rag_engine.close()


if __name__ == "__main__":
    main()