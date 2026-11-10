#!/usr/bin/env python3
"""Write a relocatable ZIP with fixed metadata from a privacy-checked release stage."""
from pathlib import Path
import shutil
import stat
import sys
import zipfile


def archive(stage, output):
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in [stage, *sorted(stage.rglob("*"))]:
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("nonregular ZIP input")
            name = stage.name + ("/" + path.relative_to(stage).as_posix() if path != stage else "")
            directory = stat.S_ISDIR(mode)
            info = zipfile.ZipInfo(name + ("/" if directory else ""), (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = ((stat.S_IFDIR | 0o755) if directory else (stat.S_IFREG | 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            if directory:
                zipped.writestr(info, b"")
            else:
                info.file_size = path.stat().st_size
                with path.open("rb") as source, zipped.open(info, "w") as destination:
                    shutil.copyfileobj(source, destination, 1 << 20)


if __name__ == "__main__":
    archive(Path(sys.argv[1]), Path(sys.argv[2]))
