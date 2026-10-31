Raw evidence is packaged in [evidence.tar.gz](evidence.tar.gz) to keep the PR file list reviewable. It contains the complete original phase snapshot, including raw JSON/logs, reproduction sources/fixtures and their original SHA manifests. `archive-members.sha256.json` hashes every payload member. Extract into an empty directory with `tar -xzf evidence.tar.gz`, then verify those SHA256 values with Python/hashlib. Summaries below remain inline. No measurement or source input was changed; no executable, library or model weights are included.

# Resumed cold-prefill qualification

Four complete boots (stock, MLP, attention, final stock), five exact prompt sizes and three completions per size: all 60 completions have zero cached tokens, exactly 16 output tokens and matching token/output hashes across every arm. Configured context remains 131072; the largest input is 130816, leaving 256 positions for output and speculative lookahead. This does not qualify the separately retained 131056-input baseline failure.

Both stock and MLP boots were completed before an interruption. Attention was restarted with its complete request history, then the final stock boot ran. The pause is disclosed; one bracket is not a confidence interval. Global auxiliary telemetry from the resumed phase covers only attention and final stock, while all earlier primary JSON timings and server logs remain available. No compilation overlapped the serving measurements.

The table uses the first request's TTFT at each size and the geometric mean of bracketing stock values. Positive delta means slower. These small observations do not establish a causal prefill speedup; they screen for a substantial regression at these settings. Cache retention was disabled identically in every arm, with the same 0.5 GiB reserve. Decode qualification uses the separate 256-token runs in the parent receipt.

| Exact prompt tokens | Stock before TTFT (s) | Candidate TTFT (s) | Stock after TTFT (s) | Delta |
|---|---:|---:|---:|---:|
| 2048 | 0.833417 | 0.842896 | 0.841499 | +0.650% |
| 8192 | 3.439917 | 3.475910 | 3.473965 | +0.550% |
| 32768 | 15.590521 | 15.719897 | 15.710568 | +0.444% |
| 65536 | 36.093146 | 36.171023 | 36.142504 | +0.147% |
| 130816 | 90.759845 | 90.810114 | 90.745606 | +0.063% |
