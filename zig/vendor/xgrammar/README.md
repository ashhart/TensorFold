# Vendored: XGrammar 0.2.8

Upstream: https://github.com/mlc-ai/xgrammar, tag `v0.2.8`, commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`.

This folder holds the C++ core from the 0.2.8 source distribution: `cpp/`, `include/` and `3rdparty/` (DLPack's header and
picojson). The build files, tests and Python bindings are left out. **Local patches: none.** Every file is byte-identical
to the upstream release, so an update means replacing the folder with the next release's same subset.

It is upstream code we do not maintain, so `tools/zig/lean_check.py` exempts exactly this prefix (`VENDORED`). Our own
code around it (`zig/src/core/grammar/`, `zig/build/grammar.zig`) is checked like any other source.

Licence: Apache License 2.0, Copyright (c) 2024 by XGrammar Contributors (`LICENSE` and `NOTICE` here). DLPack: Apache
License 2.0 (`3rdparty/dlpack/LICENSE`). picojson: BSD-2-Clause (text in `3rdparty/picojson/picojson.h`). See also the
repository's `THIRD_PARTY_NOTICES.md` and `NOTICE`.
