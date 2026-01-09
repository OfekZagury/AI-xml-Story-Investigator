import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from rag_implementations.msft_graphrag import MsftGraphRAG
from utils.data_loader import load_xml_data
from rag_implementations.naive_rag import NaiveRAG
from rag_implementations.neo4j_graphrag import Neo4jGraphRAG

load_dotenv()
current_dir = Path(__file__).parent


def print_separator():
    print("-" * 50)


def validate_environment():
    """Validate required environment variables."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file or export the key.")
        sys.exit(1)


def load_story_data(xml_path: Path) -> List[Dict[str, Any]]:
    """Load and validate XML story data."""
    print(f"Using: {xml_path}")
    messages = load_xml_data(str(xml_path))
    print(f"Successfully loaded {len(messages)} messages.")
    return messages


def get_user_strategy_choice() -> str:
    """Display menu and get user's RAG strategy choice."""
    print_separator()
    print("Choose your RAG Strategy:")
    print("1. Naive RAG (In-Memory, Simple, Fast)")
    print("2. Microsoft GraphRAG (Batch Processing, Global Summaries)")
    print("3. Neo4j GraphRAG (Database-backed, Graph+Vector)")
    return input("Enter number (1-3): ").strip()


def group_messages_by_chapter(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group messages by chapter for context preservation."""
    chapters = defaultdict(list)
    for msg in messages:
        chapters[msg['chapter']].append(msg)
    return chapters


def build_chapter_text(chapter_msgs: List[Dict[str, Any]], include_title: bool = True) -> str:
    """Build full chapter text from messages."""
    chapter_text = ""
    if include_title:
        chapter_title = chapter_msgs[0]['chapter_title']
        chapter_text = f"Chapter: {chapter_title}\n"

    for msg in chapter_msgs:
        chapter_text += f"[{msg['timestamp']}] {msg['sender_name']}: {msg['body']}\n"
    return chapter_text


def initialize_naive_rag(messages: List[Dict[str, Any]]) -> NaiveRAG:
    """Initialize Naive RAG with message data."""
    print("\nInitializing Naive RAG...")
    rag_engine = NaiveRAG()
    print("Ingesting data into memory...")

    chapters = group_messages_by_chapter(messages)

    for chapter_id, chapter_msgs in chapters.items():
        chapter_text = build_chapter_text(chapter_msgs, include_title=False)
        rag_engine.add_message(chapter_text)

    print(f"Ingested {len(chapters)} chapters (automatically chunked).")
    return rag_engine


def initialize_msft_graphrag(messages: List[Dict[str, Any]]) -> Tuple[MsftGraphRAG, str]:
    """Initialize Microsoft GraphRAG with message data."""
    print("\nInitializing Microsoft GraphRAG...")
    rag_engine = MsftGraphRAG()
    print("Preparing input files...")

    chapters = group_messages_by_chapter(messages)

    for chapter_id, chapter_msgs in chapters.items():
        chapter_title = chapter_msgs[0]['chapter_title']
        chapter_text = build_chapter_text(chapter_msgs)
        filename = f"chapter_{chapter_id}_{chapter_title}.txt"
        rag_engine.add_document(chapter_text, filename=filename)

    # Indexing step
    print("\nIMPORTANT: Microsoft GraphRAG requires an indexing step.")
    should_index = input("Run indexing now? (y/n) [Required if this is the first run]: ").lower()
    if should_index == 'y':
        rag_engine.run_indexing()
    else:
        print("Skipping indexing. (Only do this if you already indexed this EXACT data before).")

    # Search method selection
    search_method = input(
        "Search Method - 'global' (Summary) or 'local' (Specific details)? [default: global]: "
    ).strip()
    if not search_method:
        search_method = "global"

    return rag_engine, search_method


def initialize_neo4j_graphrag(messages: List[Dict[str, Any]]) -> Neo4jGraphRAG:
    """Initialize Neo4j GraphRAG with message data."""
    print("\nInitializing Neo4j GraphRAG...")

    # Check credentials
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Error: Missing Neo4j credentials in .env file.")
        sys.exit(1)

    # Connect
    try:
        rag_engine = Neo4jGraphRAG(neo4j_uri, neo4j_user, neo4j_password)
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    print("\nNeo4j Status: Connected.")

    # Ingestion decision
    should_ingest = input(
        "Do you want to ingest/re-index the data? (y/n) [Choose 'n' if data is already in DB]: "
    ).lower()

    if should_ingest == 'y':
        ingest_data_to_neo4j(rag_engine, messages)
    else:
        print("Skipping ingestion. Using existing data in Neo4j.")

    return rag_engine


def ingest_data_to_neo4j(rag_engine: Neo4jGraphRAG, messages: List[Dict[str, Any]]):
    """Ingest message data into Neo4j knowledge graph."""
    print("Preparing data for Knowledge Graph ingestion...")

    chapters = group_messages_by_chapter(messages)
    total_chapters = len(chapters)
    print(f"Found {total_chapters} chapters. Starting graph extraction process...")

    for i, (chapter_id, chapter_msgs) in enumerate(chapters.items(), 1):
        chapter_title = chapter_msgs[0]['chapter_title']
        chapter_text = build_chapter_text(chapter_msgs)

        print(f"--> Processing Chapter {i}/{total_chapters}: '{chapter_title}'")
        rag_engine.add_document(chapter_text)

    print("\nIngestion Complete! The Knowledge Graph is ready.")


def initialize_rag_strategy(choice: str, messages: List[Dict[str, Any]]) -> Tuple[Any, Optional[str]]:
    """Initialize the chosen RAG strategy.

    Returns:
        tuple: (rag_engine, search_method or None)
    """
    if choice == '1':
        return initialize_naive_rag(messages), None
    elif choice == '2':
        return initialize_msft_graphrag(messages)
    elif choice == '3':
        return initialize_neo4j_graphrag(messages), None
    else:
        raise ValueError("Invalid choice")


def run_qa_loop(rag_engine, search_method: Optional[str] = None):
    """Run the interactive question-answering loop."""
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
            if search_method and hasattr(rag_engine, 'answer_question'):
                answer = rag_engine.answer_question(question, method=search_method)
            else:
                answer = rag_engine.answer_question(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error retrieving answer: {e}")


def cleanup_resources(rag_engine, choice: str):
    """Cleanup resources based on strategy."""
    if choice == '3' and hasattr(rag_engine, 'close'):
        rag_engine.close()


def main():
    print_separator()
    print("Welcome to the Multi-Strategy RAG System")
    print_separator()

    # Validate environment
    validate_environment()

    # Load data
    xml_path = current_dir / "whodunit_sydney.xml"
    try:
        messages = load_story_data(xml_path)
    except Exception as e:
        print(f"Error loading XML: {e}")
        sys.exit(1)

    # Get user choice and initialize strategy
    choice = get_user_strategy_choice()
    try:
        rag_engine, search_method = initialize_rag_strategy(choice, messages)
    except ValueError:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")
        sys.exit(1)

    # Run Q&A loop
    run_qa_loop(rag_engine, search_method)

    # Cleanup
    cleanup_resources(rag_engine, choice)


if __name__ == "__main__":
    main()
