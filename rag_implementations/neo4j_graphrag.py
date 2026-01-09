import os
import json
from typing import List, Dict, Any
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from utils.llm_client import LLMClient


class Neo4jGraphRAG:
    """
    A RAG implementation using Neo4j as the Knowledge Graph store.
    It combines Vector Search (for finding relevant texts) with
    Graph capabilities (storing entities and relationships).
    """

    def __init__(self, uri: str, username: str, password: str):
        """
        Connect to Neo4j and initialize necessary components.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # We use the official library's embedding wrapper
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")

        # We use our own LLM client for the extraction logic
        self.llm_client = LLMClient()

        # Initialize the database schema (Indexes)
        self._initialize_schema()

    def _initialize_schema(self):
        """
        Creates a Vector Index on the 'Document' nodes if it doesn't exist.
        This allows us to perform cosine similarity searches later.
        """
        index_name = "document_embeddings"
        dim = 1536  # OpenAI text-embedding-3-small dimension

        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (d:Document) ON (d.embedding)
        OPTIONS {{indexConfig: {{
         `vector.dimensions`: {dim},
         `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            with self.driver.session() as session:
                session.run(query)
            print("[Neo4j] Vector index initialized.")
        except Exception as e:
            print(f"[Neo4j] Warning initializing index: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:
        """
        Splits text into smaller chunks to ensure context fits in LLM window.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to find a newline or period to cut cleanly, rather than mid-word
            if end < len(text):
                # Look for the last newline in the current window
                last_newline = text.rfind('\n', start, end)
                if last_newline != -1 and last_newline > start + chunk_size // 2:
                    end = last_newline + 1  # Include the newline

            chunks.append(text[start:end])
            start = end - overlap  # Move pointer back by overlap amount
        return chunks

    def add_document(self, text: str, metadata: str = ""):  # Added metadata param just in case
        """
        Ingests a document by CHUNKING it first.
        Each chunk becomes a separate Document node with its own embedding.
        """
        # 1. Split the massive chapter into smaller chunks
        chunks = self._chunk_text(text, chunk_size=800, overlap=50)
        print(f"[Neo4j] Split text into {len(chunks)} chunks. Processing...")

        import time

        for i, chunk in enumerate(chunks):
            print(f"  > Processing Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")

            # A. Embedding
            try:
                embedding = self.embedder.embed_query(chunk)
            except Exception as e:
                print(f"    [Error] Embedding failed: {e}")
                continue

            # B. Graph Extraction (LLM)
            # ask the LLM to extract entities just from this specific chunk
            extraction_prompt = f"""
            Analyze the following story segment.
            Extract key entities (Person, Location, Item, Event) and relationships.
            Output ONLY Cypher `MERGE` statements.
            Link entities to the Document node `d`.

            Text segment:
            {chunk[:3500]} 
            """

            try:
                cypher_response = self.llm_client.get_completion(extraction_prompt)
                cypher_statements = cypher_response.replace("```cypher", "").replace("```", "").strip().split("\n")
            except Exception as e:
                print(f"    [Error] Extraction failed: {e}")
                continue

            # C. Save to DB (Retry Logic)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self.driver.session() as session:
                        # Create a Node for this specific CHUNK
                        create_doc_query = """
                        CREATE (d:Document {
                            text: $text, 
                            embedding: $embedding, 
                            chapter_info: $meta,
                            chunk_index: $index,
                            timestamp: datetime()
                        })
                        RETURN elementId(d) as doc_id
                        """
                        result = session.run(create_doc_query, text=chunk, embedding=embedding, meta=metadata, index=i)
                        doc_id = result.single()["doc_id"]

                        # Connect entities to this Chunk Node
                        for statement in cypher_statements:
                            statement = statement.strip()
                            if not statement or "//" in statement: continue

                            try:
                                full_query = f"""
                                MATCH (d:Document) WHERE elementId(d) = $doc_id
                                {statement}
                                """
                                session.run(full_query, doc_id=doc_id)
                            except:
                                pass  # Ignore bad cypher

                    # Success
                    break
                except Exception as e:
                    print(f"    [Retry {attempt + 1}] Connection error: {e}")
                    time.sleep(2)

    def answer_question(self, question: str) -> str:
        """
        Retrieves relevant context using Vector Search and generates an answer.
        """
        # 1. Retrieve using VectorRetriever
        # We search for the top 3 most relevant chapters/documents based on vector similarity
        retriever = VectorRetriever(
            driver=self.driver,
            index_name="document_embeddings",
            embedder=self.embedder,
            return_properties=["text"]
        )

        print(f"[Neo4j] Retrieving context for: {question}")
        results = retriever.search(query_text=question, top_k=3)

        # Extract and concatenate the text from the results
        context_text = ""
        for result in results.items:
            context_text += f"---\n{result.content}\n"

        if not context_text:
            return "No relevant information found in the graph."

        # 2. Generate Answer - RESTRUCTURED PROMPT
        # the instruction after reading long context blocks.
        prompt = f"""
        USER QUESTION: {question}

        INSTRUCTIONS:
        Answer the question above based ONLY on the provided context below and cite the context that supports your answer.

        CONTEXT FROM DATABASE:
        <context>
        {context_text}
        </context>

        YOUR ANSWER:
        """

        # Limit prompt length
        if len(prompt) > 3000:
            raise ValueError(f"Query exceeds 3000 character limit (current: {len(prompt)} characters)")

        return self.llm_client.get_completion(prompt)

    def close(self):
        """Closes the driver connection."""
        self.driver.close()