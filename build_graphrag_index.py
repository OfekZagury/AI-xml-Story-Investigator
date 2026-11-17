"""Rebuild the Microsoft GraphRAG index in graphrag_index/ from the story XML.

    python build_graphrag_index.py            # needs GEMINI_API_KEY (see graphrag_index/settings.yaml)

The index is committed to the repository so the live demo can query it right away.
Indexing makes a few hundred LLM calls once (a few cents with Gemini Flash).
"""
import argparse
import asyncio
import re
from pathlib import Path

from dotenv import load_dotenv

from utils.data_loader import load_xml_data
from utils.story_text import build_chapter_text, group_messages_by_chapter

ROOT = Path(__file__).parent


def write_input_files(index_root: Path, xml_path: Path) -> int:
    """One text file per chapter, the same chapter text the other strategies use."""
    input_dir = index_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("*.txt"):
        old.unlink()

    chapters = group_messages_by_chapter(load_xml_data(str(xml_path)))
    for chapter_id, chapter_msgs in chapters.items():
        name = f"chapter_{chapter_id}_{chapter_msgs[0]['chapter_title']}.txt"
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        (input_dir / name).write_text(build_chapter_text(chapter_msgs), encoding="utf-8")
    return len(chapters)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "graphrag_index", help="GraphRAG workspace")
    parser.add_argument("--xml", type=Path, default=ROOT / "whodunit_sydney.xml", help="story file")
    args = parser.parse_args()
    load_dotenv()

    count = write_input_files(args.root, args.xml)
    print(f"Wrote {count} chapter files to {args.root / 'input'}")

    import graphrag.api as api
    from graphrag.config.load_config import load_config

    config = load_config(args.root)
    results = asyncio.run(api.build_index(config=config))
    failed = [r for r in results if r.errors]
    for result in results:
        print(f"  {result.workflow}: {'FAILED' if result.errors else 'ok'}")
    if failed:
        raise SystemExit(f"{len(failed)} workflow(s) failed: {failed[0].errors}")
    print("Index ready.")


if __name__ == "__main__":
    main()
