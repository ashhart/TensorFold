"""Build hook: TENSORFOLD_NATIVE_BUNDLE=DIR puts the native engine in a platform wheel; unset, nothing changes."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sysconfig

HERE = Path(__file__).resolve().parent
MAGIC = {"macosx": (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"), "linux": (b"\x7fELF",)}    # Mach-O 64/universal, ELF


def platform_tag() -> str:
    """TENSORFOLD_NATIVE_PLATFORM (a release wheel's tag), else this machine's platform for a local build."""

    return os.environ.get("TENSORFOLD_NATIVE_PLATFORM") or sysconfig.get_platform().replace("-", "_").replace(".", "_")


def bundle_files(bundle: Path, platform: str) -> list[Path]:
    """The files below the bundle's folders; refuses a missing, non-executable or other-platform binary."""

    binary = bundle / "bin" / "tensorfold-native"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"TENSORFOLD_NATIVE_BUNDLE: {binary} is missing or not executable")
    system = next((name for name in MAGIC if name in platform), "")
    with open(binary, "rb") as file:
        if file.read(4) not in MAGIC.get(system, ()):
            raise SystemExit(f"TENSORFOLD_NATIVE_BUNDLE: {binary} is not a {platform} executable")
    found = (path.relative_to(bundle) for path in bundle.rglob("*") if path.is_file())
    return sorted(path for path in found if len(path.parts) > 1 and path.parts[0] != "__pycache__")


def main() -> None:
    from setuptools import setup

    bundle = os.environ.get("TENSORFOLD_NATIVE_BUNDLE", "")
    if not bundle:
        setup()
        return
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:                                   # setuptools before 70.1
        from wheel.bdist_wheel import bdist_wheel

    source, platform = Path(bundle).resolve(), platform_tag()
    files = bundle_files(source, platform)

    class BuildNative(build_py):
        def run(self) -> None:
            super().run()
            target = Path(self.build_lib) / "tensorfold" / "native"
            for path in files:
                (target / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / path, target / path)
            shutil.copy2(HERE / "zig" / "gate.json", target / "gate.json")

    class PlatformWheel(bdist_wheel):
        def get_tag(self) -> tuple[str, str, str]:
            return "py3", "none", platform              # any Python 3 on this platform: the binary links no libpython

    class NativeDistribution(Distribution):
        def has_ext_modules(self) -> bool:
            return True                                 # the package, binary included, installs as one platlib tree

    setup(distclass=NativeDistribution, cmdclass={"build_py": BuildNative, "bdist_wheel": PlatformWheel})


if __name__ == "__main__":
    main()
