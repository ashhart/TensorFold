"""Assemble the shared Windows packager's closed runtime overlay from freshly bound inputs."""
import json
from pathlib import Path
import shutil

from runtime_sources import digest


def create(repo, work, xgrammar, grammar, bridge_manifest):
    overlay = work / "runtime-overlay"
    (overlay / "bin").mkdir(parents=True)
    (overlay / "lib").mkdir()
    (overlay / "LICENSES").mkdir()
    legal = []

    def license_file(source, name, origin):
        target = overlay / "LICENSES" / ("windows-" + name + ".txt")
        shutil.copyfile(source, target)
        legal.append({"path": target.relative_to(overlay).as_posix(),
                      "sha256": digest(target), "source": origin})
        return target.relative_to(overlay).as_posix()

    own = license_file(repo / "LICENSE", "bridge-Apache-2.0",
                       "https://github.com/ashhart/TensorFold/blob/" + bridge_manifest["source_revision"] + "/LICENSE")
    cuda = license_file(work / "bridge/bundle/licenses/NVIDIA/cuda_cudart/LICENSE",
                        "CUDA-runtime", "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json")
    cublas = license_file(work / "bridge/bundle/licenses/NVIDIA/libcublas/LICENSE",
                          "cuBLASLt", "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json")
    grammar_legal = [
        license_file(xgrammar / "LICENSE", "xgrammar-Apache-2.0",
                     "https://github.com/mlc-ai/xgrammar/blob/97787376faee5ed8466cfad57c99855e4ce2f6aa/LICENSE"),
        license_file(xgrammar / "NOTICE", "xgrammar-NOTICE",
                     "https://github.com/mlc-ai/xgrammar/blob/97787376faee5ed8466cfad57c99855e4ce2f6aa/NOTICE"),
        license_file(xgrammar / "3rdparty/dlpack/LICENSE", "dlpack-Apache-2.0",
                     "https://github.com/dmlc/dlpack/blob/bbd2f4d32427e548797929af08cfe2a9cbb3cf12/LICENSE"),
        license_file(xgrammar / "3rdparty/picojson/picojson.h", "picojson",
                     "https://github.com/mlc-ai/xgrammar/blob/97787376faee5ed8466cfad57c99855e4ce2f6aa/3rdparty/picojson/picojson.h"),
    ]
    # Preserve picojson's actual copyright/license block, rather than distributing the implementation as a notice.
    pico = overlay / grammar_legal[-1]
    text = pico.read_text(encoding="utf-8")
    end = text.find("*/")
    if not text.startswith("/*") or end < 0 or "Copyright" not in text[:end]:
        raise ValueError("picojson license block changed")
    pico.write_text(text[:end + 2] + "\n", encoding="utf-8")
    legal[-1]["sha256"] = digest(pico)
    zig_root = Path(shutil.which("zig")).resolve().parent
    zig_legal = []
    for relative, name in [("LICENSE", "zig-MIT"), ("lib/libcxx/LICENSE.TXT", "libcxx"),
                           ("lib/libcxxabi/LICENSE.TXT", "libcxxabi"),
                           ("lib/libunwind/LICENSE.TXT", "libunwind"),
                           ("lib/libc/mingw/COPYING", "mingw")]:
        zig_legal.append(license_file(zig_root / relative, name,
                                     "https://ziglang.org/download/0.17.0/"))
    libraries = []
    for source, relative, version, origin, licenses in [
        (work / "bridge/bundle/tf_cublaslt_bridge.dll", "bin/tf_cublaslt_bridge.dll", "ABI-1",
         "https://github.com/ashhart/TensorFold/tree/" + bridge_manifest["source_revision"]
         + "/tools/cuda/cublaslt_bridge", [own, cuda]),
        (work / "bridge/bundle/cublasLt64_13.dll", "bin/cublasLt64_13.dll", "13.10.0.20",
         "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/"
         "libcublas-windows-x86_64-13.10.0.20-archive.zip", [cublas]),
        (grammar, "lib/tfgrammar.dll", "xgrammar-v0.2.8",
         "https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa",
         grammar_legal + zig_legal),
    ]:
        target = overlay / relative
        shutil.copyfile(source, target)
        libraries.append({"path": relative, "sha256": digest(target), "version": version,
                          "source": origin, "licenses": licenses})
    inventory = {"schema": 1, "target": "windows-x86_64", "libraries": libraries}
    (overlay / "WINDOWS-RUNTIMES.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (overlay / "WINDOWS-RUNTIME-NOTICES.txt").write_text(
        "TensorFold Windows 5090 preview, built from this source commit.\n"
        "The bridge uses static CUDA runtime and static MSVC CRT linkage; arithmetic flags and ABI are preserved.\n"
        "NVIDIA cuBLASLt 13.10.0.20 is retained unchanged from its pinned official public archive.\n"
        "xgrammar v0.2.8, its exact dlpack submodule and TensorFold's existing grammar shim are source-built.\n"
        "The licenses directory retains the applicable project, NVIDIA, grammar and compiler-runtime notices.\n"
        "Install a compatible NVIDIA R620-or-newer driver providing nvcuda.dll and the driver's CUDA runtime provider.\n"
        "Windows OS DLLs and NVIDIA driver/provider DLLs are not bundled or replaced.\n"
        "cuBLASLt contains an NVRTC dynamic-loader candidate; optional JIT/provider paths are not model-qualified here.\n"
        "This is a CPU-built preview: fresh-head GPU load, tools, exactness, cache and performance checks remain unrun.\n"
        "The earlier cache candidate's small highest-thinking warm-code decode regression remains unresolved.\n",
        encoding="utf-8")
    (work / "runtime-legal-inputs.private.json").write_text(json.dumps(legal, indent=2) + "\n")
    return overlay
