# Integrated CUDA weight-upload check

On integrated Linux CUDA devices, Qwen's file-backed MLX tensors read into one owned pinned stage and upload from that stage. Nemotron's direct-I/O read slots are owned pinned stages too. Completion failure retains those bytes through the existing copy-source quarantine, even after file handles and checkpoint maps close. Constructed vectors and discrete devices keep their ordinary upload paths. No layout, arithmetic or kernel changes are involved (#2043).

The Qwen reader uses the existing direct-I/O adapter. Filesystems that refuse direct I/O use its buffered fallback, still reading into the same owned buffer. Qwen reserves one 64 MiB slot plus alignment at open, so later pieces never grow its pinned allocation. Nemotron retains four 64 MiB slots and its existing overlapping reads. Qwen waits for each piece before reusing its one stage. These are ownership choices, not measured speed claims.

`tf-cuda-weight-load qwen|nemotron MODEL time|hash|control OUTPUT` opens the actual family engine on device 0. It requires an integrated sm_121 GPU, refuses an existing output, and writes a JSON receipt after load. The Qwen receipt supports dense MLX checkpoints; Nemotron disables MTP and graphs and reserves 4,096 rows. Qwen uses its ordinary engine initialization without creating a served context. There is no HTTP server.

- `time`: initialization and family load seconds, process peak RSS and CUDA device/pinned allocation peaks before readback.
- `hash`: separate process, SHA-256 of final device-resident weights using pinned asynchronous DtoH followed by completion.
- `control`: changes one final device byte after load, then hashes. The checkpoint is never written.

Qwen hashes its zero-initialized weight arena through the recorded end of weight allocations, excluding later scratch. Nemotron hashes every named weight allocation. Padding is included: unstable A/A bytes make the receipt inconclusive, rather than being omitted. Hashes establish equality of those final buffers, not a separate arithmetic or served-token qualification.

Build with pinned Zig and the same real CUDA images/compiler for both builds:

```sh
zig build cuda-weight-load -Dtarget=aarch64-linux-gnu -Dsm=121 -Dnvcc="$NVCC" --prefix "$INSTALL"
```

The tool installs to `$INSTALL/bin/tf-cuda-weight-load`. Compile-only checks with inert images cannot produce a runnable build. Attach source, compiler, executable, kernel-image and checkpoint revisions/hashes to a device receipt.

Use separate processes for timing and hashing. Pin `TF_LOAD_PREFETCH=0` in both builds, discard warm loads, then measure A1/B1/B2/A2 for each family. These timings are a warm sequence with read-ahead off; record the cache state and label filesystem residency unmeasured when it was not checked. Also measure one A/B pair with `TF_LOAD_PREFETCH` unset in both builds, using the default read-ahead, and record its timings and memory separately. Don't clear the page cache or change clocks between arms; record whether the cache was warm.

Record every raw timing and separate memory peak; do not add unrelated peaks together. Require stable base A/A hashes, matching branch hashes and a differing planted control. Compare load improvement with the observed A/A spread. Inspect the default read-ahead pair before declaring no gain: a positive pair needs its own A/A and A/B/B/A before it establishes a benefit. If neither policy shows a gain beyond its observed A/A spread, stop and discard the upload replacement.

Before landing, inspect peak-memory changes and run the applicable cold 32K and served-token checks. Device performance and model equality are unchecked until the device receipts return. Host tests use a recording driver with delayed DMA, file/mmap disagreement, an unaligned piece tail, early-free detection, producer/completion failures and the unchanged discrete path.
