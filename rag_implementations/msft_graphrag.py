import os
import sys
import subprocess
import re
from typing import List

class MsftGraphRAG:
    """
    A wrapper around the Microsoft GraphRAG CLI.
    """

    def __init__(self, root_dir: str = "./rag_workspace"):
        """
        Initialize the workspace for MS GraphRAG.
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, "input")

        # Locate the graphrag CLI executable
        scripts_dir = os.path.dirname(sys.executable)
        self.cli_path = os.path.join(scripts_dir, "graphrag")

        if os.name == 'nt' and not self.cli_path.endswith('.exe'):
            self.cli_path += ".exe"

        # check if CLI exists
        if not os.path.exists(self.cli_path):
            raise FileNotFoundError(f"Could not find graphrag CLI at {self.cli_path}. Did you install 'graphrag'?")

        # 1. Create directory structure
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        # 2. Initialize GraphRAG if settings.yaml is missing
        settings_path = os.path.join(self.root_dir, "settings.yaml")
        if not os.path.exists(settings_path):
            print(f"[MsftGraphRAG] Initializing workspace in {self.root_dir}...")

            # using graph init directly
            subprocess.run([self.cli_path, "init", "--root", self.root_dir], check=True)

        # 3. Always ensure settings are configured for our needs (Model & API Key)
        self._configure_settings(settings_path)

    def _configure_settings(self, settings_path: str):
        """
        Updates settings.yaml to use gpt-5.
        Now includes a check to prevent double-replacement corruption.
        """
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'model: gpt-5' in content:
            print("[MsftGraphRAG] Settings already configured for gpt-5. Skipping.")
            return

        print("[MsftGraphRAG] Updating settings.yaml to use gpt-5...")

        # Replace the default API key placeholder
        if 'api_key: ${GRAPHRAG_API_KEY}' in content:
            content = content.replace('api_key: ${GRAPHRAG_API_KEY}', 'api_key: ${OPENAI_API_KEY}')

        # FORCE override models to gpt-5
        content = content.replace('model: gpt-4-turbo-preview', 'model: gpt-5')
        content = content.replace('model: gpt-4', 'model: gpt-5')

        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def add_document(self, text: str, filename: str = "doc.txt"):
        """
        Saves a text chunk to the input directory.
        """
        clean_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        file_path = os.path.join(self.input_dir, clean_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {clean_filename}")

    def run_indexing(self):
        """
        Runs the indexing process using the CLI executable.
        """
        print("\n[MsftGraphRAG] Starting Indexing... (This creates entities and communities)")
        print("Note: This might take 1-3 minutes depending on data size.")
        try:
            subprocess.run([self.cli_path, "index", "--root", self.root_dir], check=True)
            print("[MsftGraphRAG] Indexing completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[MsftGraphRAG] Indexing failed: {e}")

    def answer_question(self, question: str, method: str = "global") -> str:
        """
        Queries the RAG system.
        """
        # Limit prompt length
        if len(question) > 3000:
            raise ValueError(f"Query exceeds 3000 character limit (current: {len(question)} characters)")

        print(f"Querying GraphRAG ({method} search)...")

        try:
            result = subprocess.run(
                [
                    self.cli_path, "query",
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

            output = result.stdout

            # Parse output
            match = re.search(r'SUCCESS:.*Search Response:\s*(.*)', output, re.DOTALL)
            if match:
                return match.group(1).strip()

            return f"(Raw Output:) {output}"

        except Exception as e:
            return f"Query execution failed: {e}"