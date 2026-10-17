# Drama 3 source and qualification

The family and both backends come from drowzeys' PRs #939 and #956. Shared frame scheduling,
the native adapter, FP32 PTH mapping and merge repairs are maintainer adaptations of both implementations.
The speech protocol, restricted PTH reader, allocation cap, module-marker checks and direct-I/O stager
come from plotarmordev's #915–#927 stack. The staged BF16 GEMV candidate comes from that stack too.
The candidate is exposed only through a diagnostic probe. Serving does not select it.

The development oracle follows SGLang-Omni revision 416a70c0. `oracle_prompt.py` retains its Apache-2.0
SPDX and source notice. `oracle_model.py` is the original oracle's model portion, split without
changing its arithmetic. `codec_convert.py` remains an independent offline FP32 schema reference.
Serving reads the checkpoint and `codec.pth` in Zig. See `THIRD_PARTY_NOTICES.md` for the original
model, codec and reference-source notices. No model or codec weights are included.

`oracles_927/*.zig.txt` are byte-for-byte copies of #927's historical diagnostic programs.
They are reference data, not executable tests of the merged family. They retain their independent
arithmetic references, trace names and fixture definitions. They require the original #927 adapters
to execute unchanged. The merged tool uses #939/#956's teacher-forced/layer/encoder checks and
`--replay 1` for eager/graph, poison/reset, codec-chunk and interleaved-state controls.

`--speed-probe 1` compares the staged GEMV with the merged CUDA baseline on the named projection.
A successful projection probe is insufficient to promote a serving implementation. Re-run the
complete frame, PCM, WAV, reference, chunk-seam and resource gates on the exact merged source.
The historical #927 same-audio receipts do not qualify the different FP32 codec or sampler.

Original commit identities must be recovered from the upstream commits before creating authored
commits or Co-authored-by trailers. The supplied local commit lists identify hashes and subjects,
not verified contributor email addresses. Do not invent those addresses.
