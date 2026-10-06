# Native runtime files

`bin/tensorfold-native --version` and `--help` need no model, driver, Python package or repository checkout.
The macOS binary targets the Apple M1 CPU instruction set and macOS 13 or later. Metal is supplied by macOS.
The Linux binaries target baseline x86_64 or aarch64 with glibc 2.28 or later. The NVIDIA driver supplies `libcuda.so.1`.

At this source revision, Nemotron's Metal kernels and draft vocabulary are embedded in the executable.
It does not load an adjacent Nemotron coop metallib. `lib/` is reserved for future adjacent runtime libraries.
CUDA `.cu` fatbins are embedded at build time from the qualified directory supplied to the release build.

Inference also needs a model directory or an existing Hugging Face cache, including config, tokenizer, template
and weight files. No model download occurs at startup. Flash Next Metal needs the captured directory selected by
`TF_FLASHNEXT_DUMP`. CUDA Nemotron needs `aot.json` and its `cubins/` directory, selected by
`TENSORFOLD_CUDA_KERNELS` or found at `share/tensorfold/cuda/sm<capability>/` relative to the executable.
These model-specific capture assets are not created by the release build. An optional capture root can be
packaged with `-Ddist-cuda-aot=DIR`. The release owner must qualify it for the intended GPU.

Flash Next speed-up settings also name the separately built MCDMA dynamic library and peer settings.
That library is not part of these archives. The server can read an API key file when requested.
None of those paths requires an installed Python package, though capture generation currently uses Python tools.
Archives marked `host-only` are CPU verification artifacts and cannot serve CUDA inference.
