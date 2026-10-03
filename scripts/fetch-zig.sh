#!/usr/bin/env bash
# Reuse the pinned stable compiler or stage its verified release archive.
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(cat .zig-version)
if [[ -x .zig-toolchain/zig ]]; then
    installed=$(.zig-toolchain/zig version)
    if [[ "$installed" == "$version" ]]; then
        echo "Zig $installed is already staged."
        exit 0
    fi
    echo "Existing .zig-toolchain uses $installed; move it aside before staging $version." >&2
    exit 1
fi
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) platform=aarch64-macos ;;
    *) echo "The native Metal build requires Apple Silicon macOS." >&2; exit 1 ;;
esac
if system_zig=$(command -v zig) && [[ "$system_zig" == /* && "$("$system_zig" version)" == "$version" ]]; then
    mkdir -p .zig-toolchain
    ln -s "$system_zig" .zig-toolchain/zig
    echo "Using installed Zig $version: $system_zig"
    exit 0
fi
asset="zig-$platform-$version"
read -r expected_digest expected_path < .zig-archive.sha256
if [[ "$expected_path" != "build/toolchains/$asset.tar.xz" || ! "$expected_digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Update .zig-archive.sha256 together with .zig-version before downloading." >&2
    exit 1
fi
mkdir -p build/toolchains
if [[ ! -f "build/toolchains/$asset.tar.xz" ]]; then
    curl -fSL --retry 3 "https://ziglang.org/download/$version/$asset.tar.xz" -o "build/toolchains/$asset.tar.xz.part"
    mv "build/toolchains/$asset.tar.xz.part" "build/toolchains/$asset.tar.xz"
fi
shasum -a 256 --check .zig-archive.sha256
tar -xf "build/toolchains/$asset.tar.xz" -C build/toolchains
actual=$("build/toolchains/$asset/zig" version)
[[ "$actual" == "$version" ]]
mv "build/toolchains/$asset" .zig-toolchain
echo "Staged Zig $actual. Build with .zig-toolchain/zig build."
