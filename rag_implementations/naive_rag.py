import numpy as np
from typing import List, Dict, Tuple
from utils.llm_client import LLMClient


class NaiveRAG:
    """
    A simple implementation of Retrieval-Augmented Generation (RAG).
    It handles text chunking, embedding generation, vector similarity search,
    and answer generation using an LLM.
    """

    def __init__(self):
        """
        Initializes the RAG system.
        Uses LLMClient for both embeddings and text generation.
        """
        self.llm_client = LLMClient()
        # In-memory vector store: List of dictionaries {'text': str, 'vector': np.array}
        self.vector_store: List[Dict] = []

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Splits a large text into smaller overlapping chunks.

        Args:
            text (str): The input text.
            chunk_size (int): The maximum number of characters per chunk.
            overlap (int): The number of characters to overlap between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a vector embedding for a given text using OpenAI's embedding model.
        """
        text = text.replace("\n", " ")  # Normalize newlines
        try:
            # Accessing the internal openai client from LLMClient
            response = self.llm_client.client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            vector = response.data[0].embedding
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.array([])

    def add_message(self, text: str):
        """
        Processes a document: chunks it, embeds each chunk, and stores it in memory.
        """
        print("Chunking and embedding chapter...")
        chunks = self.chunk_text(text)

        for chunk in chunks:
            vector = self.get_embedding(chunk)
            if vector.size > 0:
                self.vector_store.append({
                    "text": chunk,
                    "vector": vector
                })
        print(f"Stored {len(chunks)} chunks in vector store.")

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Finds the most relevant chunks for a query using Cosine Similarity.
        """
        query_vector = self.get_embedding(query)
        if query_vector.size == 0:
            return []

        scored_chunks = []

        for item in self.vector_store:
            # Calculate Cosine Similarity
            # Note: OpenAI embeddings are normalized, so dot product equals cosine similarity
            score = np.dot(query_vector, item['vector'])
            scored_chunks.append((score, item['text']))

        # Sort by score descending (highest similarity first)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Return the top K text chunks
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def answer_question(self, question: str) -> str:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks based on the question.
        2. Construct a prompt with context.
        3. Get answer from LLM.
        """
        print(f"Analyzing question: {question}")

        # 1. Retrieve
        relevant_chunks = self.find_relevant_chunks(question)
        if not relevant_chunks:
            return "I couldn't find any relevant information in the documents."

        # --- DEBUG PRINT START ---
        print("\n--- DEBUG: Retrieved Chunks ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"[{i + 1}] {chunk[:100]}...")  # beginning of chunk
        print("--- DEBUG END ---\n")
        # --- DEBUG PRINT END ---

        # 2. Augment (Construct Prompt)
        context_text = "\n\n---\n\n".join(relevant_chunks)

        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.
        If the answer is not in the context, say you don't know.
        Context:
        {context_text}
        User Question: 
        {question}
        Answer:
        """

        # 3. Generate
        return self.llm_client.get_completion(prompt)


# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    pass