import os
import tempfile
import unittest

from utils.data_loader import load_xml_data


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <participants>
    <person id="p1" name="Alice" role="Detective"/>
    <person id="p2" name="Bob" role="Witness"/>
    <group id="g1" name="Harbour Crew"/>
  </participants>
  <chapter id="c1" title="The Call">
    <message id="m1" ts="2024-05-01T21:14:00">
      <sender ref="p1"/>
      <receiver ref="p2"/>
      <body>Where were you last night?</body>
    </message>
    <message id="m2" ts="2024-05-01T21:16:00">
      <sender ref="p2"/>
      <receiver ref="g1"/>
      <body>At the harbour, ask anyone.</body>
      <attachment type="image" file="pier.jpg" exif="2024-05-01T20:58:00"/>
    </message>
  </chapter>
  <chapter id="c2" title="The Alibi">
    <message id="m3" ts="2024-05-02T08:00:00">
      <sender ref="g1"/>
      <receiver ref="p1"/>
      <body>He left before ten.</body>
    </message>
  </chapter>
</story>
"""


class TestXmlDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.xml_file_path = os.path.join(self.test_dir.name, "sample_story.xml")
        with open(self.xml_file_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_XML)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_parses_all_messages_across_chapters(self):
        messages = load_xml_data(self.xml_file_path)

        self.assertIsInstance(messages, list)
        self.assertEqual([m["id"] for m in messages], ["m1", "m2", "m3"])
        self.assertEqual([m["chapter"] for m in messages], ["c1", "c1", "c2"])
        self.assertEqual(messages[2]["chapter_title"], "The Alibi")

    def test_message_fields_and_sender_names(self):
        first, second, third = load_xml_data(self.xml_file_path)

        self.assertEqual(first["timestamp"], "2024-05-01T21:14:00")
        self.assertEqual(first["sender"], "p1")
        self.assertEqual(first["receiver"], "p2")
        self.assertEqual(first["sender_name"], "Alice")
        self.assertEqual(first["body"], "Where were you last night?")
        # Group chats resolve to the group name instead of "Unknown"
        self.assertEqual(third["sender_name"], "Harbour Crew")

    def test_attachments_are_parsed(self):
        messages = load_xml_data(self.xml_file_path)

        self.assertNotIn("attachment", messages[0])
        self.assertEqual(
            messages[1]["attachment"],
            {"type": "image", "file": "pier.jpg", "exif": "2024-05-01T20:58:00"},
        )

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_xml_data("non_existent_file.xml")


if __name__ == "__main__":
    unittest.main()
