"""Cache-only success cannot stand in for full token equality or observed retention."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cache_client', Path(__file__).with_name('reproduce-cache.py'))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def row(phase, cached, sha='b' * 64):
    state = 'unverified' if cached is None else 'reused' if cached else 'cold'
    return {'phase': phase, 'cached_tokens': cached, 'cache_state': state,
            'prompt_tokens': 4122, 'tokens': 8, 'complete': True,
            'output_sha256': 'a' * 64, 'token_sha256': sha}


class Exactness(unittest.TestCase):
    def test_cache_qualification_cannot_replace_full_token_evidence(self):
        rows = [row('first', 0, None), row('repeated', 4096, None)]
        self.assertTrue(client.bench_cache.qualify(rows, 4122, True, True))
        self.assertFalse(client.qualified(rows, 4122, True, False))

    def test_real_reuse_requires_positive_edited_first_counter(self):
        self.assertFalse(client.qualified([row('first', 0), row('repeated', 4096)], 4122, True, True))
        self.assertTrue(client.qualified([row('first', 4096), row('repeated', 4096)], 4122, True, True))

    def test_wrong_hash_and_short_fingerprint_cannot_pass(self):
        self.assertFalse(client.qualified([row('first', 0), row('repeated', 4096, 'c' * 64)], 4122, True, False))
        self.assertFalse(client.qualified([row('first', 0, 'b' * 12)], 4122, False, False))

    def test_fresh_control_rejects_retention(self):
        self.assertFalse(client.qualified([row('first', 4096)], 4122, False, False))
        self.assertTrue(client.qualified([row('first', 0)], 4122, False, False))

    def test_early_completion_is_not_an_eight_token_reply(self):
        observed = row('first', 0)
        observed['tokens'] = 4
        self.assertFalse(client.qualified([observed], 4122, False, False))

    def test_unknown_edited_first_counter_is_unverified(self):
        self.assertFalse(client.qualified([row('first', None), row('repeated', 4096)], 4122, True, True))


if __name__ == '__main__':
    unittest.main()
