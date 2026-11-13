"""An ELF libtfvideo.so exports exactly the four tf_video_* functions, keeps a planted table local, has no TEXTREL."""
import struct
import sys
from pathlib import Path

WANT = {'tf_video_open', 'tf_video_frames', 'tf_video_close', 'tf_video_version'}
SHT_SYMTAB, SHT_DYNAMIC, SHT_DYNSYM = 2, 6, 11
DT_TEXTREL, DT_FLAGS, DF_TEXTREL = 22, 30, 0x4


def sections(data: bytes) -> list[tuple]:
    """(type, offset, size, link, entsize) for each section of a little-endian ELF64 file."""
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        raise SystemExit('not a little-endian ELF64 file')
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', data, 0x3a)
    out = []
    for i in range(shnum):
        _, kind, _, _, off, size, link, _, _, ent = struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize)
        out.append((kind, off, size, link, ent))
    return out


def symbols(data: bytes, secs: list[tuple], kind: int) -> dict[str, tuple[int, int]]:
    """Defined symbols of the given table: name -> (binding, section index)."""
    out = {}
    for t, off, size, link, ent in secs:
        if t != kind:
            continue
        strtab = secs[link][1]
        for at in range(off + ent, off + size, ent):
            name, info, _, shndx = struct.unpack_from('<IBBH', data, at)
            if shndx:
                end = data.index(b'\0', strtab + name)
                out[data[strtab + name:end].decode()] = (info >> 4, shndx)
    return out


def textrel(data: bytes, secs: list[tuple]) -> bool:
    """True when .dynamic has DT_TEXTREL or DF_TEXTREL in DT_FLAGS."""
    for t, off, size, _, _ in secs:
        if t == SHT_DYNAMIC:
            for at in range(off, off + size, 16):
                tag, val = struct.unpack_from('<qQ', data, at)
                if tag == DT_TEXTREL or (tag == DT_FLAGS and val & DF_TEXTREL):
                    return True
    return False


def main() -> int:
    failed = 0
    for arg in sys.argv[1:]:
        data = Path(arg).read_bytes()
        secs = sections(data)
        exported = {n for n, (bind, _) in symbols(data, secs, SHT_DYNSYM).items() if bind in (1, 2)}
        planted = symbols(data, secs, SHT_SYMTAB).get('tfvideo_planted_table')
        problems = [f'exports {sorted(exported ^ WANT)} beyond or short of the four'] if exported != WANT else []
        if planted is None or planted[0] != 0:
            problems.append(f'the planted table is not a local symbol: {planted}')
        if textrel(data, secs):
            problems.append('text relocations')
        machine = {62: 'x86_64', 183: 'aarch64'}.get(struct.unpack_from('<H', data, 0x12)[0], 'other')
        verdict = 'FAIL' if problems else 'PASS'
        print(f'{verdict} {machine} {arg}: {len(exported)} exported' + ''.join(f'; {p}' for p in problems))
        failed += bool(problems)
    return 1 if failed or len(sys.argv) < 2 else 0


if __name__ == '__main__':
    sys.exit(main())
