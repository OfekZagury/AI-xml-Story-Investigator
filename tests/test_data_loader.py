import unittest
import os
import tempfile

# Note: The following line will fail right now because we haven't created the file and function yet
# We'll implement this in the next step in src/data_loader.py
# from src.data_loader import load_xml_data

class TestXmlDataLoader(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary XML file before each test.
        """
        self.test_dir = tempfile.TemporaryDirectory()
        self.xml_file_path = os.path.join(self.test_dir.name, "sample_data.xml")

        # Sample XML content that simulates the structure of articles or documents
        self.xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <library>
        <document id="101">
        <title>GraphRAG Introduction</title>
        <content>GraphRAG combines knowledge graphs with LLMs.</content>
        <author>Alice</author> 
        </document> 
        <document id="102"> 
        <title>Python Testing</title> 
        <content>TDD ensures code quality.</content> 
        <author>Bob</author> 
        </document> 
        </library> 
        """

        with open(self.xml_file_path, "w", encoding="utf-8") as f:
            f.write(self.xml_content)

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        self.test_dir.cleanup()

    def test_load_xml_valid_parsing(self):
        """
        Test that the parser correctly extracts data from a valid XML file.
        """
        # Local import so that the test does not crash immediately when run before execution, but only inside the function
        try:
            from src.data_loader import load_xml_data
        except ImportError:
            self.fail("Could not import 'load_xml_data'. Did you implement it yet?")

        #Act
        results = load_xml_data(self.xml_file_path)

        # Assert
        self.assertIsInstance(results, list, "Output should be a list")
        self.assertEqual(len(results), 2, "Should parse exactly 2 documents")

        # Checking the first document
        first_doc = results[0]
        self.assertEqual(first_doc['title'], "GraphRAG Introduction")
        self.assertIn("knowledge graphs", first_doc['content'])
        self.assertEqual(first_doc['id'], "101")

        # Checking the second document
        second_doc = results[1]
        self.assertEqual(second_doc['author'], "Bob")

    def test_load_xml_file_not_found(self):
        """
        Test that the function raises FileNotFoundError for non-existent files.
        """
        try:
            from src.data_loader import load_xml_data
        except ImportError:
            self.fail("Could not import 'load_xml_data'.")

        with self.assertRaises(FileNotFoundError):
            load_xml_data("non_existent_file.xml")

if __name__ == '__main__':
    unittest.main()