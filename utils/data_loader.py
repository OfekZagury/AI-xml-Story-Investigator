import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


def load_xml_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse XML story data containing messages organized in chapters.
    Handles XML namespaces and attribute-based metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found at: {file_path}")

    tree = ET.parse(file_path)
    root = tree.getroot()

    # 1. Define the Namespace
    # The XML has xmlns="urn:whodunit:sms:1", so we must map it.
    ns = {'ns': 'urn:whodunit:sms:1'}

    messages = []
    characters = {}

    # 2. Parse Participants (Corrected tag name and attribute access)
    # XML uses <person>, not <character>
    # XML uses attributes (name="..."), not child tags (<name>...</name>)
    for person in root.findall('.//ns:person', ns):
        char_id = person.get('id')
        characters[char_id] = {
            'name': person.get('name'),
            'role': person.get('role')
        }

    # Also add groups to characters dict so we don't get 'Unknown' for group messages
    for group in root.findall('.//ns:group', ns):
        group_id = group.get('id')
        characters[group_id] = {
            'name': group.get('name'),
            'role': 'Group Chat'
        }

    # 3. Parse messages from all chapters
    for chapter in root.findall('.//ns:chapter', ns):
        chapter_id = chapter.get('id')
        chapter_title = chapter.get('title')

        for message in chapter.findall('ns:message', ns):
            # Helper to safely get sender/receiver refs
            sender_elem = message.find('ns:sender', ns)
            receiver_elem = message.find('ns:receiver', ns)
            body_elem = message.find('ns:body', ns)

            sender_ref = sender_elem.get('ref') if sender_elem is not None else "unknown"
            receiver_ref = receiver_elem.get('ref') if receiver_elem is not None else "unknown"

            msg_data = {
                'id': message.get('id'),
                'timestamp': message.get('ts'),
                'chapter': chapter_id,
                'chapter_title': chapter_title,
                'sender': sender_ref,
                'receiver': receiver_ref,
                'body': body_elem.text if body_elem is not None else "",
                'sender_name': characters.get(sender_ref, {}).get('name', 'Unknown'),
                'receiver_name': characters.get(receiver_ref, {}).get('name', 'Unknown')
            }

            # Handle optional attachments
            attachment = message.find('ns:attachment', ns)
            if attachment is not None:
                msg_data['attachment'] = {
                    'type': attachment.get('type'),
                    'file': attachment.get('file'),
                    'exif': attachment.get('exif')
                }
                # Audio attachments carry their metadata in `meta` instead of `exif`
                if attachment.get('meta'):
                    msg_data['attachment']['meta'] = attachment.get('meta')

            messages.append(msg_data)

    return messages

