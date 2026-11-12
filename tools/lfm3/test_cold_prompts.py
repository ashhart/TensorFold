"""The generated cold-prompt files must reproduce the committed hashes."""
import json
import unittest

import cold_prompts


class ColdPromptTests(unittest.TestCase):
    def test_regenerated_files_match_recorded_hashes(self):
        wanted = json.loads(cold_prompts.SHA256.read_text())
        self.assertEqual(cold_prompts.mismatches(cold_prompts.files(), wanted), [])

    def test_a_changed_file_is_reported(self):
        wanted = json.loads(cold_prompts.SHA256.read_text())
        produced = cold_prompts.files()
        produced["text-2048.ids.json"] += b" "
        self.assertEqual(cold_prompts.mismatches(produced, wanted), ["text-2048.ids.json"])


if __name__ == "__main__":
    unittest.main()
