import os
import subprocess
import sys
import re
from typing import List


class MsftGraphRAG:
    """
    A wrapper around the Microsoft GraphRAG CLI.

    This implementation handles:
    1. Workspace initialization.
    2. Configuration tweaking.
    3. Indexing execution.
    4. Query execution with log parsing.
    """

    def __init__(self, root_dir: str = "./rag_workspace"):
        """
        Initialize the workspace for MS GraphRAG.
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, "input")

        # 1. Create directory structure
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        # 2. Initialize GraphRAG if settings.yaml is missing
        settings_path = os.path.join(self.root_dir, "settings.yaml")
        if not os.path.exists(settings_path):
            print(f"[MsftGraphRAG] Initializing workspace in {self.root_dir}...")
            # We run the init command
            subprocess.run([sys.executable, "-m", "graphrag.index", "--init", "--root", self.root_dir], check=True)

        # 3. Always ensure settings are configured for our needs (Model & API Key)
        self._configure_settings(settings_path)

    def _configure_settings(self, settings_path: str):
        """
        Updates settings.yaml to use gpt-4o-mini.
        GraphRAG defaults to gpt-4 which is slow and expensive for indexing.
        """
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace the default API key placeholder with the env variable reference
        if 'api_key: ${GRAPHRAG_API_KEY}' in content:
            content = content.replace('api_key: ${GRAPHRAG_API_KEY}', 'api_key: ${OPENAI_API_KEY}')

        # FORCE override models to gpt-4o-mini to save cost/time
        # We replace the default model configuration
        # Note: This is a simple string replace. For complex config, use a YAML parser.
        content = content.replace('model: gpt-4-turbo-preview', 'model: gpt-4o-mini')
        content = content.replace('model: gpt-4', 'model: gpt-4o-mini')

        # Write back the updated settings
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def add_document(self, text: str, filename: str = "doc.txt"):
        """
        Saves a text chunk to the input directory.
        Sanitizes filename to avoid OS errors.
        """
        # Clean filename (keep only alphanumerics, dots, dashes)
        clean_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

        file_path = os.path.join(self.input_dir, clean_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {clean_filename}")

    def run_indexing(self):
        """
        Runs the indexing process. This must be called after adding all documents.
        """
        print("\n[MsftGraphRAG] Starting Indexing... (This creates entities and communities)")
        print("Note: This might take 1-3 minutes depending on data size.")
        try:
            subprocess.run([sys.executable, "-m", "graphrag.index", "--root", self.root_dir], check=True)
            print("[MsftGraphRAG] Indexing completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[MsftGraphRAG] Indexing failed: {e}")

    def answer_question(self, question: str, method: str = "global") -> str:
        """
        Queries the RAG system.

        Args:
            question (str): The user query.
            method (str): 'global' (summarization/holistic) or 'local' (specific facts).
                          Global is better for "What happened?", Local for "Who is X?".
        """
        print(f"Querying GraphRAG ({method} search)...")

        # Decide which CLI command to run based on version/method
        # GraphRAG usually uses the same command with a flag
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "graphrag.query",
                    "--root", self.root_dir,
                    "--method", method,
                    "--query", question
                ],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.returncode != 0:
                return f"Error in CLI: {result.stderr}"

            # Parse the output to hide system logs
            # The CLI typically ends with: "SUCCESS: <Method> Search Response: \n <The Answer>"
            output = result.stdout

            # Try to catch the clean answer
            match = re.search(r'SUCCESS:.*Search Response:\s*(.*)', output, re.DOTALL)
            if match:
                return match.group(1).strip()

            # If regex fails, return the tail of the output as fallback
            return f"(Raw Output) {output[-2000:]}"

        except Exception as e:
            return f"Query execution failed: {e}"