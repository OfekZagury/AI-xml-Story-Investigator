"""Turns parsed story messages into chapter texts used by every RAG strategy."""
from collections import defaultdict
from typing import Any, Dict, List


def group_messages_by_chapter(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group messages by chapter for context preservation (keeps the story order)."""
    chapters = defaultdict(list)
    for msg in messages:
        chapters[msg['chapter']].append(msg)
    return chapters


def format_message(msg: Dict[str, Any]) -> str:
    """One line per message: time, sender → receiver, body and any attachment metadata.

    Attachment metadata (photo EXIF times, audio recordings) are clues, so they stay in the text.
    """
    receiver = msg.get('receiver_name')
    line = f"[{msg['timestamp']}] {msg['sender_name']}"
    if receiver and receiver != 'Unknown':
        line += f" → {receiver}"
    line += f": {msg['body']}"

    attachment = msg.get('attachment')
    if attachment:
        details = attachment.get('exif') or attachment.get('meta')
        line += f" [{attachment.get('type')} attachment {attachment.get('file')}"
        line += f", {details}]" if details else "]"
    return line


def build_chapter_text(chapter_msgs: List[Dict[str, Any]], include_title: bool = True) -> str:
    """Build full chapter text from messages."""
    chapter_text = ""
    if include_title:
        chapter_title = chapter_msgs[0]['chapter_title']
        chapter_text = f"Chapter: {chapter_title}\n"

    for msg in chapter_msgs:
        chapter_text += format_message(msg) + "\n"
    return chapter_text
