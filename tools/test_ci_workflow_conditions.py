"""Keep pull-request branch labels out of CI security decisions."""
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class WorkflowConditionTests(unittest.TestCase):
    def test_landing_train_branches_cannot_skip_pull_request_checks(self):
        for name in ("app.yml", "no-gpu.yml", "zig-fmt.yml"):
            with self.subTest(workflow=name):
                workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
                jobs = workflow["jobs"]
                self.assertTrue(jobs)
                triggers = workflow.get("on", workflow.get(True))
                self.assertIn("pull_request", triggers)
                for job_name, job in jobs.items():
                    condition = str(job.get("if", ""))
                    self.assertNotIn("github.head_ref", condition, job_name)
                    self.assertNotIn("github.event.pull_request.head.ref", condition, job_name)


if __name__ == "__main__":
    unittest.main()
