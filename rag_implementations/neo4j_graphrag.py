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

    def add_document(self, text: str):
        """
        Ingests a document into the graph.
        1. Calculates the vector embedding.
        2. Uses an LLM to extract entities and relationships (Graph Construction).
        3. Saves everything to Neo4j in a single transaction.
        """
        print(f"[Neo4j] Processing document ({len(text)} chars)...")

        # 1. Generate Embedding
        # The embedder returns a list of embeddings, we take the first one
        embedding = self.embedder.embed_query(text)

        # 2. Extract Graph Elements using LLM
        # We ask the LLM to write Cypher for us. This is a "Text-to-Cypher" approach.
        extraction_prompt = f"""
        You are a graph database expert. Analyze the following story segment.
        Extract key entities (Person, Location, Item, Event) and their relationships.

        Output ONLY a list of Cypher `MERGE` statements to represent these facts.
        Always link the entities to a central Document node using the variable `d`.

        Rules:
        1. Do NOT create the Document node `d` (it is passed as a parameter).
        2. Use `MERGE` to avoid duplicates.
        3. Keep names simple (e.g., 'Marcus', 'USB').
        4. Output format: plain text, one statement per line. No markdown.

        Text to analyze:
        {text[:4000]} 
        """

        # Get Cypher from LLM
        cypher_response = self.llm_client.get_completion(extraction_prompt)

        # Clean the response (remove code blocks if LLM added them)
        cypher_statements = cypher_response.replace("```cypher", "").replace("```", "").strip().split("\n")

        # 3. Write to Database
        with self.driver.session() as session:
            # First, create the Document node
            create_doc_query = """
            CREATE (d:Document {text: $text, embedding: $embedding, timestamp: datetime()})
            RETURN id(d) as doc_id
            """
            result = session.run(create_doc_query, text=text, embedding=embedding)
            doc_id = result.single()["doc_id"]

            # Now run the generated Cypher statements to attach entities to this document
            # We match the document 'd' first so the LLM's queries can attach to it
            for statement in cypher_statements:
                statement = statement.strip()
                if not statement or "//" in statement: continue

                try:
                    # We wrap the LLM's statement to provide the context 'd'
                    full_query = f"""
                    MATCH (d:Document) WHERE id(d) = $doc_id
                    {statement}
                    """
                    session.run(full_query, doc_id=doc_id)
                except Exception as e:
                    # LLM might generate invalid Cypher occasionally, we skip and log
                    print(f"  [Neo4j] Skipped invalid Cypher: {e}")

        print("[Neo4j] Document and graph entities saved.")

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
        Answer the question above based ONLY on the provided context below.

        CONTEXT FROM DATABASE:
        <context>
        {context_text}
        </context>

        YOUR ANSWER:
        """

        return self.llm_client.get_completion(prompt)

    def close(self):
        """Closes the driver connection."""
        self.driver.close()