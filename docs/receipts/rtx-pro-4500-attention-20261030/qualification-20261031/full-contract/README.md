# Complete numerical run; backend limit-record gate fails

The UNFILTERED CUDA program finished every numerical/control/standalone phase, then exited1 on gate5. **This is not a full backend pass.**

```
SUMMARY 465 kernels: 465 pass gates 1-4, 0 fail, 8 known findings; 322 controls: 322 red as planted, 0 missed; 223 limit problems; gates 5132023 ms, limits 1130269 ms
```

All465 registered kernels pass gates1-4, including independent MLP bit/FP64 cases and both MT2 replacement variants. All322 numerical planted controls are caught, zero missed. Eight declared known findings remain. Both MT2 forms inherit main's tf_q35_attn_merge NaN-key behavior: a row can remain finite where idealFP64 givesNaN. It is reported KNOWN; no new NaN-rule fix is claimed.

The223 limit problems are168 missing static/compiler records plus55 live usage mismatches:46 recorded as zero and9 stale nonzero records. The changed launches ALL pass their own live checks:

```
PASS limits live tf_q35_attn_rows8_mt2: 128 threads launched, the driver allows 128; 249 registers, 36992 + 0 shared bytes (recorded 249, 36992)
PASS limits live tf_q35_attn_rows8_mt2_whole: 128 threads launched, the driver allows 128; 248 registers, 36992 + 0 shared bytes (recorded 248, 36992)
PASS limits live qmm_group MLP128: 256 threads launched, the driver allows 256; 54 registers, 0 + 26880 shared bytes (recorded 54, 0)
```

No missing/static or failed live line names these three new launches. The diagnosis lists every missing/stale record and shows unchanged baseline pending-record source files. An unmodified-main limits-only control is recorded separately when available; no numerical cases are rerun or waived to hide this backend failure. PRs stay drafts while the global limit-record issue remains unresolved.

Elapsed 6263.4s. Maximum observed processRSS is 16.251GiB (2-second HWM sampling), under the unchanged24GiB container cap. Late cgroup sampling records zeroOOM/max events. A configured budget on this31.24GiB desktop is not a minimum-RAM claim, and no isolated memory-reduction percentage is inferred.

This is one COMBINED integration test build on main94b90c9d: independent MLP and attention patches plus a common five-file TEST-ONLY fixture-lifetime fix. It is not an unmodified or single-contribution runner. Each replacement compares its own kernel against stock on identical inputs; served performance is measured independently. No context-headroom, stride or GDN-fusion change is present.

The common test patch releases completed Rig buffers/host copies after each kernel/control and scopes registered owners/modules so existing destructors run before standalone suites. Math, shapes, name-derived seeds, controls and bounds remain unchanged. Host build171/171steps,139/139 fresh reported tests passes with cached steps retained; lean424 inherited/0new. Exact build/component/combined patches, source/binarySHA, toolchain/image and all full/partial logs are archived.

The first142-second attempt stopped because the queued launcher still held the pre-fix desktop-status reader. It has no SUMMARY and is retained. The corrected monitor reran the same binary with the original admission deadline and background preemption; hardware/profile/clock/power/runtime installations were unchanged.

Physical coverage is one RTX PRO4500 SM120/82SM,5900X,Linux,driver615.78.08,nvcc13.3.73,Zig0.17.0. Other GPUs/backends and previous-release/standard-server comparisons are unrun. Earlier compile-only/sanitizer receipts retain their separate scope.

[evidence.tar.gz](evidence.tar.gz) contains all full/partial runtime logs, build logs/metadata, exact diagnosis, reproduction drivers and component/combined source patches. Extract in an empty directory and verify archive-members.sha256.json. No model, executable, library or toolchain binaries are included.

The main-only limit control completed: {"candidate_and_main_live_failure_lines_identical": true, "candidate_and_main_missing_record_sets_identical": true, "live_mismatch_count": 55, "main_base": "94b90c9db65dcb71bbc39715901d6fb1f00407d0", "main_only_exit_code": 1, "main_only_summary": ["LIMITS_SUMMARY main-only: 223 problems"], "missing_static_record_count": 168, "only_baseline_test_change": "Five-line CLI hook calls existing contract_tests.live; all production kernels and registry entries are unmodified.", "total_limit_problems_both": 223, "unmodified_main_numerical_full_suite_not_rerun": true}
