Raw evidence is packaged in [evidence.tar.gz](evidence.tar.gz) to keep the PR file list reviewable. It contains the complete original phase snapshot, including raw JSON/logs, reproduction sources/fixtures and their original SHA manifests. `archive-members.sha256.json` hashes every payload member. Extract into an empty directory with `tar -xzf evidence.tar.gz`, then verify those SHA256 values with Python/hashlib. Summaries below remain inline. No measurement or source input was changed; no executable, library or model weights are included.

# Alternating MLP decode noise check

In response to issue1700, six independent fresh-calibration stock/candidate/stock groups (18 native server processes). Same single-contribution native binary68553364dc2a4e50bc609505453486aa6f4d51d4f643864e9dd555bb3855b191, independently rebased onto main94b90c9db65dcb71bbc39715901d6fb1f00407d0. TF_CUDA_TEST_MLP128=0/1/0 in every group. Canonical greedy code/prose,256 output tokens, seven seeded repetitions and one warmup per prompt per process. All18 process token fingerprints match. No GDN or attention change in this binary.

**Geometric mean gain: +0.825%.** Approximate95% Student-t interval on six group log ratios: **+0.754% to +0.896%**. All six groups are positive (exact two-sided sign test p=0.03125). No request/token pseudo-replicates: the group is the statistical unit. The interval assumes approximately normal independent group log gains; one machine/time window limits generalization.

| Group | Greedy geometric mean gain | Code gain | Prose gain | Code baseline drift | Prose baseline drift |
|---|---:|---:|---:|---:|---:|
| 1 | +0.900% | +1.103% | +0.697% | +0.028% | -0.764% |
| 2 | +0.707% | +0.731% | +0.683% | -0.791% | -0.688% |
| 3 | +0.846% | +0.838% | +0.854% | -0.084% | -0.059% |
| 4 | +0.792% | +0.811% | +0.772% | -0.061% | -0.054% |
| 5 | +0.857% | +0.869% | +0.844% | +0.023% | +0.025% |
| 6 | +0.849% | +0.875% | +0.823% | +0.008% | +0.027% |

Use each candidate median against the geometric mean of its bracketing stock medians, then geometric mean over the two prompts. Baseline drift columns are after/before, retained rather than normalized away. Repeated baselines, raw client values, output hashes, startup override confirmation, manifest, calibration boundaries and one-second GPU telemetry are included.

Settings preserved: configured131072 context, vision/tools/schema/reasoning support, FP8 KV, int8 prefill, BF16 DFlash2,8GiB host prompt cache,parallel1,default200W,headless compute profile,upstream0.5GiB reserve. Only speed requests disable thinking. No CPU build overlaps performance timing. Exact checkpoint/tokenizer/drafter/library identities are in the parent receipt and each manifest. Version1.1.6 is recipe compatibility metadata, not the actual source revision.

A loginctl session disappeared between listing and inspection, interrupting the second group on the initial attempt. That incomplete group is excluded as a whole and was restarted in full. The first complete group was retained. All partial outputs/logs and the before-run plan remain attached under interrupted-attempt. The wrapper retries that transient policy read but still stops on actual profile change. No timing-based exclusion or sequential stopping was used; all six predeclared complete groups are reported.

This is a new classic-greedy protocol, kept separate from community/long-context/feature and earlier reserve0 receipts. It demonstrates a small repeatable gain on this4500, not every workload or GPU. No physical5090/PRO6000 ratio. Full CUDA contract qualification is separately in progress; these timing results do not replace its SUMMARY gate. Source/fixtures and exact analysis are attached; no weights, libraries or executables are repackaged.
