# Host rendering after compact builder handles

This compares original base `f350cd3ea253a042e205746928271f9d8f71a82d`, earlier pointer-based PR `d33e5e769d7a4bb801cc39e0a32400488cb89777`, and revised implementation `8d00b490f5ddf45721546e9b9816a083a46e5fa2` on Apple M5 Pro, 48 GiB RAM, macOS 27.0, Zig 0.17.0 ReleaseSafe. It is a host template-render benchmark, not served TTFT or decode.

The [driver](run.py), frozen at `2f6f3382f` before measurements, selects four cases from the unchanged, checksum-pinned corpora. Every case runs all six revision permutations in a fixed order, giving six fresh processes per revision. All binaries are built before timing. Each process performs two excluded warmups and three timed renders using the unchanged earlier compatibility harness. Compilation, input creation and output validation are outside the timed region. Other task-owned compiler/test workloads had completed before timing began.

All 72 processes and 360 renders pass full Jinja byte equality and canonical provenance identity checks. Of these, 144 renders are excluded warmups and 216 are timed. No runs were added or replaced after inspecting results. The same 240 MiB combined compilation/render-scratch cap remains in force; all selected cases complete.

The table uses the median of six per-process medians. Percentages use unrounded values. Negative time change means faster. These are descriptive results, with no confidence interval or equivalence claim; the small short-request change is not evidence of a material slowdown.

| Workload | Base ms | Earlier PR ms | Revised ms | Revised vs base | Revised vs earlier PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 messages | 0.05358 | 0.06054 | 0.05381 | +0.43% | -11.11% |
| 2,001 messages, no tools | 16.52515 | 18.05583 | 15.94325 | -3.52% | -11.70% |
| 2,001 messages, 157 synthetic schemas | 16.43798 | 17.57704 | 15.69854 | -4.50% | -10.69% |
| 64 cycles, 8 KiB serialized object arguments | 10.06496 | 2.26796 | 2.13719 | -78.77% | -5.77% |

The filler slowdown observed with the earlier larger `Value` is absent from this fixed comparison, while the large-tool render remains faster. This does not establish the same percentages for other request shapes, checkpoints, hardware or served inference. The full scratch-memory comparison and remaining peak differences are in the [layout receipt](../value-layout-followup/README.md).

[Metadata](metadata.json), [all samples](results.json), [summary](summary.json), and [independent raw-log verification](verification.json) retain source, helper, harness and binary identities. The verification separates hashes of the original pretty JSON from the compact published JSON; parsed measurement values are identical. The unchanged harness verifies complete rendered bytes every time; the summary is not inferred from build success alone.

```sh
python3 docs/receipts/template-accumulation/value-layout-timing/run.py \
  --template /tmp/deepseek-template.jinja --exact-corpus /tmp/compatibility-corpus.json \
  --allocation-corpus /tmp/allocation-corpus.json \
  --candidate 8d00b490f5ddf45721546e9b9816a083a46e5fa2 \
  --output /tmp/value-layout-timing --zig zig
```
