"""Live web demo: interrogate the murder mystery and compare RAG strategies side by side.

Runs on Streamlit Community Cloud with Google Gemini. Naive RAG builds its vector index
when the app starts; Microsoft GraphRAG queries the index committed in graphrag_index/.
"""
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------- configuration
# Hosted secrets become environment variables before the RAG modules read them.
try:
    for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # no secrets file when running locally - .env is used instead
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from rag_implementations.msft_graphrag import GraphRAGIndex  # noqa: E402
from rag_implementations.naive_rag import NaiveRAG  # noqa: E402
from utils.data_loader import load_xml_data  # noqa: E402
from utils.story_text import build_chapter_text, format_message, group_messages_by_chapter  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).parent
STORY_PATH = ROOT / "whodunit_sydney.xml"
GRAPHRAG_ROOT = ROOT / "graphrag_index"

MAX_QUESTIONS_PER_VISITOR = 10
MAX_QUESTIONS_PER_DAY = 100   # across all visitors, keeps the API budget predictable
MAX_QUESTION_CHARS = 300
ANSWER_TTL_SECONDS = 24 * 3600  # the same question is answered from cache for a day

STRATEGIES = {
    "naive": (
        "Naive RAG",
        "Splits the chats into overlapping chunks, embeds them and sends the 3 most similar chunks to the LLM.",
    ),
    "global": (
        "GraphRAG · global",
        "Microsoft GraphRAG: map-reduce over LLM-written summaries of the story's entity communities.",
    ),
    "local": (
        "GraphRAG · local",
        "Microsoft GraphRAG: the entities, relationships and messages closest to the question.",
    ),
}
MODES = {"Compare all three": list(STRATEGIES)} | {label: [key] for key, (label, _) in STRATEGIES.items()}

EXAMPLES = [
    "What happened to Liam on the ferry?",
    "Who is responsible for what happened to Liam, and what is the evidence?",
    "What do the photo and audio timestamps reveal?",
    "Why was Marcus's USB locked with a passphrase?",
    "What was in the unattended bag at the quay?",
    "Summarize the whole story in five sentences.",
]

st.set_page_config(page_title="AI XML Story Investigator · Live demo", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------- data and engines
@st.cache_data(show_spinner=False)
def story_messages():
    return load_xml_data(str(STORY_PATH))


@st.cache_resource(show_spinner="Reading the 348 messages and building the vector index…")
def naive_engine():
    rag = NaiveRAG()
    for chapter_msgs in group_messages_by_chapter(story_messages()).values():
        rag.add_message(build_chapter_text(chapter_msgs, include_title=False))
    if not rag.vector_store:
        raise RuntimeError("Embedding the story failed")
    return rag


@st.cache_resource(show_spinner="Loading the GraphRAG knowledge graph…")
def graph_engine():
    return GraphRAGIndex(GRAPHRAG_ROOT)


@st.cache_resource
def shared_store():
    """Process-wide store shared by all visitors: cached answers and the daily question counter."""
    return {"answers": {}, "day": None, "count": 0, "lock": threading.Lock()}


def reserve_daily_question():
    store = shared_store()
    with store["lock"]:
        today = datetime.now(timezone.utc).date().isoformat()
        if store["day"] != today:
            store["day"], store["count"] = today, 0
        if store["count"] >= MAX_QUESTIONS_PER_DAY:
            return False
        store["count"] += 1
        return True


def release_daily_question():
    store = shared_store()
    with store["lock"]:
        store["count"] = max(0, store["count"] - 1)


def run_strategy(strategy, question, engine, answers):
    """Returns {"answer", "context", "seconds", "error"} for one strategy (never raises).

    Runs in a worker thread, so the engine and the shared answer cache are passed in
    rather than fetched from Streamlit's caches here.
    """
    key = (strategy, " ".join(question.lower().split()))
    cached = answers.get(key)
    if cached and time.time() - cached["at"] < ANSWER_TTL_SECONDS:
        return cached | {"cached": True}

    started = time.time()
    try:
        if engine is None:
            raise RuntimeError("engine unavailable")
        if strategy == "naive":
            answer, context = engine.answer_with_context(question)
        else:
            answer, context = engine.search(question, method=strategy)
        if not answer:
            raise RuntimeError("empty answer")
        result = {"answer": answer, "context": context, "seconds": time.time() - started, "error": None}
        answers[key] = result | {"at": time.time()}
        return result
    except Exception as exc:  # shown as a friendly message, details go to the logs
        print(f"[{strategy}] failed for {question!r}: {exc!r}")
        return {"answer": None, "context": None, "seconds": time.time() - started, "error": str(exc)}


# ---------------------------------------------------------------- rendering
def render_context(strategy, context):
    if not context:
        return
    if strategy == "naive":
        label = f"Retrieved chunks ({len(context)})"
    else:
        label = "What GraphRAG used"
    with st.expander(label):
        if isinstance(context, list):
            for i, chunk in enumerate(context, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(chunk)
        else:
            st.markdown(context)


def tidy_answer(text):
    """Fit LLM markdown into a narrow column: headings become bold lines, citations turn grey."""
    text = re.sub(r"^\s*#{1,6}\s*(.+?)\s*#*\s*$", r"**\1**", text, flags=re.M)
    # GraphRAG cites its sources as [Data: Reports (3, 6); Entities (4)]
    return re.sub(r"\[Data: ([^\]]+)\]", r":gray[(\1)]", text)


def render_result(strategy, result):
    title, how = STRATEGIES[strategy]
    st.subheader(title)
    st.caption(how)
    if result["error"]:
        st.error("This strategy didn't answer this time (the AI service may be busy). Please try again in a minute.")
        return
    st.markdown(tidy_answer(result["answer"]))
    note = "cached answer" if result.get("cached") else f"{result['seconds']:.1f}s"
    st.caption(f"⏱ {note}")
    render_context(strategy, result["context"])


def render_story_browser(messages):
    chapters = group_messages_by_chapter(messages)
    titles = {cid: msgs[0]["chapter_title"] for cid, msgs in chapters.items()}
    chapter = st.selectbox(
        "Chapter",
        list(chapters),
        format_func=lambda cid: f"Chapter {cid} · {titles[cid]}",
    )
    lines = [format_message(m) for m in chapters[chapter]]
    st.text("\n".join(lines))


# ---------------------------------------------------------------- page
messages = story_messages()
people = sorted({m["sender_name"] for m in messages} - {"Unknown"})

st.title("🔎 AI XML Story Investigator")
st.markdown(
    f"A murder mystery told in **{len(messages)} SMS messages** between friends in Sydney, stored as XML. "
    "Ask anything about it and compare how different **Retrieval-Augmented Generation** strategies answer "
    "the same question. Powered by **Google Gemini**."
)
st.caption(
    "⚠️ Spoilers: questions about the ending reveal the solution. "
    "· [Source code on GitHub](https://github.com/OfekZagury/AI-xml-Story-Investigator)"
)

if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.error("The demo is not configured: no GEMINI_API_KEY found.")
    st.stop()

with st.expander(f"📱 Read the evidence: {len(group_messages_by_chapter(messages))} chapters · {', '.join(people)}"):
    render_story_browser(messages)

if "asked" not in st.session_state:
    st.session_state.asked = 0
    st.session_state.question = EXAMPLES[0]
    st.session_state.last = None

mode = st.segmented_control("Strategy", list(MODES), default="Compare all three")
picked = st.pills("Try a question", EXAMPLES, default=None, key="example")
if picked and picked != st.session_state.get("last_example"):
    st.session_state.question = picked
    st.session_state.last_example = picked

left_runs = MAX_QUESTIONS_PER_VISITOR - st.session_state.asked
with st.form("ask"):
    question = st.text_input(
        "Your question",
        key="question",
        max_chars=MAX_QUESTION_CHARS,
        placeholder="e.g. Who lied to the police, and about what?",
    )
    submitted = st.form_submit_button("Ask", type="primary", disabled=left_runs <= 0)
st.caption(
    f"Questions left in this session: {max(left_runs, 0)}. Identical questions are answered from a shared cache."
    if left_runs > 0
    else "You've used this session's questions. Thanks for trying the demo!"
)

if submitted:
    question = (question or "").strip()
    strategies = MODES.get(mode or "Compare all three", list(STRATEGIES))
    if not question:
        st.warning("Please type a question.")
    elif not reserve_daily_question():
        st.warning("The demo reached today's question limit. Please come back tomorrow.")
    else:
        st.session_state.asked += 1
        with st.spinner("Investigating… GraphRAG global search reads the whole story, so it can take ~20 seconds."):
            # Build/fetch the engines here: Streamlit's caches belong to the script thread.
            engines = {}
            for strategy in strategies:
                try:
                    engines[strategy] = naive_engine() if strategy == "naive" else graph_engine()
                except Exception as exc:
                    print(f"[{strategy}] engine start-up failed: {exc!r}")
                    engines[strategy] = None
            answers = shared_store()["answers"]
            with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
                futures = {s: pool.submit(run_strategy, s, question, engines[s], answers) for s in strategies}
                results = {s: f.result() for s, f in futures.items()}
        if all(r["error"] for r in results.values()):
            st.session_state.asked -= 1
            release_daily_question()
        st.session_state.last = {"question": question, "results": results}
        st.rerun()  # redraw the whole page so the counter above reflects this question

if st.session_state.last:
    last = st.session_state.last
    st.markdown(f"### ❓ {last['question']}")
    columns = st.columns(len(last["results"]), gap="large")
    for column, (strategy, result) in zip(columns, last["results"].items()):
        with column:
            render_result(strategy, result)

st.divider()
st.caption(
    "The repository also implements a third strategy on **Neo4j** (an LLM-built knowledge graph plus a "
    "vector index). It needs a database server, so it runs locally rather than in this demo."
)
