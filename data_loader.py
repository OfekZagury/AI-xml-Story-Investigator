import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


def load_xml_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses an XML file and returns a list of dictionaries.
    Each dictionary represents a document within the XML.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        List[Dict]: A list of documents with their fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the XML is malformed.
    """
    # A check to ensure the file exists before attempting to parse it
    # Fits the requirement from the test test_load_xml_file_not_found
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        results = []

        # Assuming each document is represented by a <document> tag
        # Fits the test test_load_xml_valid_parsing
        for doc in root.findall('document'):
            doc_data = {}

            # extracting the 'id' attribute if it exists
            if 'id' in doc.attrib:
                doc_data['id'] = doc.attrib['id']

            # extracting child elements like title, content, author, etc.
            for child in doc:
                doc_data[child.tag] = child.text.strip() if child.text else ""

            results.append(doc_data)

        return results

    except ET.ParseError as e:
        # throws an error if the XML is malformed
        raise ValueError(f"Error parsing XML file: {e}")