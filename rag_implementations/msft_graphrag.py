import asyncio
import os
import sys
import subprocess
import re
import threading
from pathlib import Path
from typing import Any, Tuple

# The committed, ready-to-query index (built with Gemini, see build_graphrag_index.py)
DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent / "graphrag_index")


class MsftGraphRAG:
    """
    A wrapper around the Microsoft GraphRAG CLI.
    """

    def __init__(self, root_dir: str = DEFAULT_ROOT):
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

        if 'model_provider: gemini' in content:
            print("[MsftGraphRAG] Using the Gemini settings in settings.yaml.")
            return

        if 'model: gpt-5' in content:
            print("[MsftGraphRAG] Settings already configured for gpt-5. Skipping.")
            return

        print("[MsftGraphRAG] Updating settings.yaml to use gpt-5...")

        # Replace the default API key placeholder
        if 'api_key: ${GRAPHRAG_API_KEY}' in content:
            content = content.replace('api_key: ${GRAPHRAG_API_KEY}', 'api_key: ${OPENAI_API_KEY}')

        # FORCE override chat models to gpt-5 (covers gpt-4, gpt-4o, gpt-4.1, gpt-4-turbo-preview ...)
        # A plain string replace would turn 'gpt-4o' into the invalid 'gpt-5o'.
        content = re.sub(r'(model:\s*)gpt-4[\w.\-]*', r'\g<1>gpt-5', content)

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

class GraphRAGIndex:
    """
    Queries an already-built GraphRAG index in-process with the graphrag Python API.

    Used by the web demo: no CLI subprocess, and the index tables are loaded once.
    """

    COMMUNITY_LEVEL = 2
    RESPONSE_TYPE = "A few short paragraphs"

    def __init__(self, root_dir: str = DEFAULT_ROOT):
        import pandas as pd
        from graphrag.config.load_config import load_config

        self.root = Path(root_dir)
        self.config = load_config(self.root)
        output = self.root / "output"
        self.entities = pd.read_parquet(output / "entities.parquet")
        self.communities = pd.read_parquet(output / "communities.parquet")
        self.community_reports = pd.read_parquet(output / "community_reports.parquet")
        self.text_units = pd.read_parquet(output / "text_units.parquet")
        self.relationships = pd.read_parquet(output / "relationships.parquet")

        # GraphRAG caches its model clients, and they stay bound to the event loop that first
        # used them. Running every search on one long-lived loop keeps concurrent searches
        # (e.g. from several web requests) from waiting forever on another loop's objects.
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="graphrag-loop", daemon=True).start()

    def search(self, question: str, method: str = "global") -> Tuple[str, str]:
        """Answers a question with global or local search; returns (answer, context summary)."""
        import graphrag.api as api

        if len(question) > 3000:
            raise ValueError(f"Query exceeds 3000 character limit (current: {len(question)} characters)")

        if method == "global":
            coroutine = api.global_search(
                config=self.config,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                community_level=self.COMMUNITY_LEVEL,
                dynamic_community_selection=False,
                response_type=self.RESPONSE_TYPE,
                query=question,
            )
        elif method == "local":
            coroutine = api.local_search(
                config=self.config,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                text_units=self.text_units,
                relationships=self.relationships,
                covariates=None,
                community_level=self.COMMUNITY_LEVEL,
                response_type=self.RESPONSE_TYPE,
                query=question,
            )
        else:
            raise ValueError(f"Unknown search method: {method}")

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        response, context = future.result(timeout=180)
        return str(response), self.describe_context(context)

    @staticmethod
    def describe_context(context: Any) -> str:
        """A short markdown summary of what the search read (reports, entities, sources...)."""
        if not isinstance(context, dict):
            return ""
        lines = []
        for name, table in context.items():
            if not hasattr(table, "empty") or table.empty:
                continue
            title_column = next((c for c in ("title", "entity", "source", "target") if c in table.columns), None)
            line = f"- **{name.replace('_', ' ').capitalize()}:** {len(table)}"
            if title_column:
                sample = [str(v) for v in table[title_column].head(6).tolist()]
                line += " (" + "; ".join(sample) + ("; …" if len(table) > 6 else "") + ")"
            lines.append(line)
        return "\n".join(lines)
