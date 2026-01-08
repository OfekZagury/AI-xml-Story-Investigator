import os
import sys
from dotenv import load_dotenv
from pathlib import Path

from rag_implementations.msft_graphrag import MsftGraphRAG
from utils.data_loader import load_xml_data
from rag_implementations.naive_rag import NaiveRAG
from rag_implementations.neo4j_graphrag import Neo4jGraphRAG

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
        from collections import defaultdict
        chapters = defaultdict(list)

        # 1. Group messages
        for mes in messages:
            chapters[mes['chapter']].append(mes)

        # 2. Feed whole chapters to the RAG
        for chapter_id, msgs in chapters.items():
            # Combine all messages in the chapter into one long string
            chapter_text = ""
            for msg in msgs:
                chapter_text += (
                    f"[{msg['timestamp']}] {msg['sender_name']}: {msg['body']}\n"
                )

            # Now we send a LARGE text (e.g., 5000 chars).
            # The NaiveRAG.chunk_text function will split this into
            # 5 overlapping chunks of 1000 chars each.
            # This preserves context!
            rag_engine.add_message(chapter_text)  # (You might want to rename this method to add_document)

        print(f"Ingested {len(chapters)} chapters (automatically chunked).")
    elif choice == '2':
        print("\nInitializing Microsoft GraphRAG...")
        rag_engine = MsftGraphRAG()

        print("Preparing input files...")

        from collections import defaultdict
        # A: prepare & add documents
        chapters_buffer = defaultdict(list)
        for msg in messages:
            chapters_buffer[msg['chapter']].append(msg)

        for chapter_id, chapter_msgs in chapters_buffer.items():
            chapter_title = chapter_msgs[0]['chapter_title']

            # Build full chapter text
            chapter_full_text = f"Chapter: {chapter_title}\n"
            for msg in chapter_msgs:
                chapter_full_text += (
                    f"[{msg['timestamp']}] {msg['sender_name']}: {msg['body']}\n"
                )

            # saving to file, filename must be unique
            filename = f"chapter_{chapter_id}_{chapter_title}.txt"
            rag_engine.add_document(chapter_full_text, filename=filename)

        # B: indexing
        print("\nIMPORTANT: Microsoft GraphRAG requires an indexing step.")
        should_index = input("Run indexing now? (y/n) [Required if this is the first run]: ").lower()
        if should_index == 'y':
            rag_engine.run_indexing()
        else:
            print("Skipping indexing. (Only do this if you already indexed this EXACT data before).")

        # C: choose search method
        search_method = input(
            "Search Method - 'global' (Summary) or 'local' (Specific details)? [default: global]: ").strip()
        if not search_method:
            search_method = "global"



    elif choice == '3':
        print("\nInitializing Neo4j GraphRAG...")
        # 1. Check environment variables
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            print("Error: Missing Neo4j credentials in .env file.")
            sys.exit(1)
        # Initialize the engine
        try:
            rag_engine = Neo4jGraphRAG(neo4j_uri, neo4j_user, neo4j_password)

        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            sys.exit(1)

        # 2. Decide whether to ingest data (Ingestion Strategy)
        # to extract entities and relationships. We only want to do this once.
        print("\nNeo4j Status: Connected.")

        should_ingest = input("Do you want to ingest/re-index the data? (y/n) [Choose 'n' if data is already in DB]: ").lower()

        if should_ingest == 'y':
            print("Preparing data for Knowledge Graph ingestion...")
            # --- Grouping Logic (Context Preservation) ---
            #  group messages by chapter so the LLM can extract entities
            # from a complete context rather than isolated messages.
            from collections import defaultdict
            chapters_buffer = defaultdict(list)

            for msg in messages:
                chapters_buffer[msg['chapter']].append(msg)

            total_chapters = len(chapters_buffer)
            print(f"Found {total_chapters} chapters. Starting graph extraction process...")

            for i, (chapter_id, chapter_msgs) in enumerate(chapters_buffer.items(), 1):
                chapter_title = chapter_msgs[0]['chapter_title']
                # Construct the full chapter text
                chapter_full_text = f"Chapter: {chapter_title}\n"
                for msg in chapter_msgs:
                    chapter_full_text += (
                        f"[{msg['timestamp']}] {msg['sender_name']}: {msg['body']}\n"
                    )

                print(f"--> Processing Chapter {i}/{total_chapters}: '{chapter_title}'")
                rag_engine.add_document(chapter_full_text)

            print("\nIngestion Complete! The Knowledge Graph is ready.")

        else:
            print("Skipping ingestion. Using existing data in Neo4j.")

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