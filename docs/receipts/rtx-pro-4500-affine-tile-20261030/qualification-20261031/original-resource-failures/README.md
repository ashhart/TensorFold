Raw evidence is packaged in [evidence.tar.gz](evidence.tar.gz) to keep the PR file list reviewable. It contains the complete original phase snapshot, including raw JSON/logs, reproduction sources/fixtures and their original SHA manifests. `archive-members.sha256.json` hashes every payload member. Extract into an empty directory with `tar -xzf evidence.tar.gz`, then verify those SHA256 values with Python/hashlib. Summaries below remain inline. No measurement or source input was changed; no executable, library or model weights are included.

# Original complete-run attempts: resource failures

These are the original unmodified full CUDA contract programs. Stock and attention exited-9 together after1990seconds, each sampled process highwater about9.5GB. The cgroup memory.events snapshot records2OOMkills. The surviving MLP runner later exited-9 after4582.6seconds at sampled highwater25,616,699,392bytes, near the24GiB container limit. That counter snapshot predates the final MLP kill; its final cgroup event count was not retained.

None produced a SUMMARY. Individual passing numerical lines and expected red planted controls do not replace full qualification. A subsequent stock-only attempt stopped fail-closed when loginctl lost a transient session between enumeration and lookup. It is retained privately as an incomplete attempt, not a numerical failure or pass.

The next comparison uses the same original source snapshots with only a shared test-fixture lifetime overlay: completion hooks release CUDA Rig device buffers and host copies between kernels. All numerical gates, seeds, shapes, controls and limits remain intact. This changes the test harness, so a future pass must disclose that overlay and its binary/source hashes. Original binaries remain preserved separately. No full pass is claimed yet.

Paths and trailing blank lines in these receipt copies are normalized; original logs remain unchanged in the local experiment workspace. No runtime feature or installed service was changed.
