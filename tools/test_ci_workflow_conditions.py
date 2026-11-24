"""Limit landing train job skips to the repository's own pull requests."""
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class WorkflowConditionTests(unittest.TestCase):
    def test_only_same_repository_landing_trains_can_skip_jobs(self):
        expected_skips = {"app.yml": 5, "no-gpu.yml": 7, "web.yml": 3, "zig-fmt.yml": 1}
        train_condition = (
            "${{ !(startsWith(github.head_ref, 'lead/train-') && "
            "github.event.pull_request.head.repo.full_name == github.repository) }}"
        )
        for name, expected_count in expected_skips.items():
            with self.subTest(workflow=name):
                workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
                jobs = workflow["jobs"]
                self.assertTrue(jobs)
                triggers = workflow.get("on", workflow.get(True))
                self.assertIn("pull_request", triggers)
                count = 0
                for job_name, job in jobs.items():
                    condition = str(job.get("if", ""))
                    if "github.head_ref" in condition or "github.event.pull_request.head.ref" in condition:
                        self.assertEqual(condition, train_condition, job_name)
                        count += 1
                self.assertEqual(count, expected_count)


if __name__ == "__main__":
    unittest.main()
