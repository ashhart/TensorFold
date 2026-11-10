"""Decode bounded CUDA containers and ELF sections without executing device or host code."""
import hashlib
import struct
import zlib
from compression import zstd

MAGIC = b"\x50\xed\x55\xba"
MAX_IMAGE = 256 << 20
MAX_TOTAL = 8 << 30


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decompress(payload):
    if payload.startswith(b"\x28\xb5\x2f\xfd"):
        decoder = zstd.ZstdDecompressor()
        result = decoder.decompress(payload, max_length=MAX_IMAGE + 1)
        if len(result) > MAX_IMAGE or not decoder.eof or any(decoder.unused_data):
            raise ValueError("incomplete, excessive or concatenated Zstandard image")
        return result, "zstd"
    return payload, "none"


def elf(data, emit):
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected ELF64 little endian")
    fields = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    _, kind, machine, _, _, phoff, shoff, flags, _, phsize, phnum, shsize, shnum, names_index = fields
    if machine != 190 or not 0 < shnum < 65535 or shsize < 64 or names_index >= shnum:
        raise ValueError("unsupported CUDA ELF geometry")
    if shoff + shsize * shnum > len(data) or phoff + phsize * phnum > len(data):
        raise ValueError("CUDA ELF table bounds")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + shsize * index) for index in range(shnum)]
    names = sections[names_index]
    if names[4] + names[5] > len(data):
        raise ValueError("CUDA ELF section-name bounds")
    names_data = data[names[4]:names[4] + names[5]]
    report = []
    for index, row in enumerate(sections):
        name_at, category, attributes, _, at, size, link, _, _, entry_size = row
        end = names_data.find(b"\0", name_at)
        if end < 0 or name_at >= len(names_data):
            if index == 0 and name_at == 0 and not names_data:
                name = ""
            else:
                raise ValueError("CUDA ELF section name")
        else:
            name = names_data[name_at:end].decode("ascii", "strict")
        if category != 8 and at + size > len(data):
            raise ValueError("CUDA ELF section payload bounds")
        entry = {"name": name, "type": category, "flags": attributes, "offset": at, "bytes": size}
        if category != 8:
            payload = data[at:at + size]
            entry["sha256"] = sha(payload)
            if attributes & 0x800:
                if size < 24:
                    raise ValueError("compressed ELF header bounds")
                method, _, decoded_size, _ = struct.unpack_from("<IIQQ", payload)
                if decoded_size > MAX_IMAGE:
                    raise ValueError("compressed ELF section limit")
                if method == 1:
                    decoder = zlib.decompressobj()
                    decoded = decoder.decompress(payload[24:], MAX_IMAGE + 1)
                    if not decoder.eof or decoder.unused_data or len(decoded) != decoded_size:
                        raise ValueError("compressed ELF zlib size mismatch")
                elif method == 2:
                    decoded, _ = decompress(payload[24:])
                    if len(decoded) != decoded_size:
                        raise ValueError("compressed ELF zstd size mismatch")
                else:
                    raise ValueError("unrecognized ELF compression")
                emit(decoded, "elf-section-" + str(index))
                entry["decoded_sha256"] = sha(decoded)
                entry["decoded_bytes"] = len(decoded)
            if category in (2, 11):
                if entry_size < 24 or link >= shnum or size % entry_size:
                    raise ValueError("CUDA ELF symbol table bounds")
        report.append(entry)
    return {"type": kind, "machine": machine, "flags": flags, "sections": report}


def iter_containers(data, emit, state=None, interleaved=False):
    state = state if state is not None else {"decoded_bytes": 0, "entries": 0}
    found, cursor = 0, 0
    while cursor < len(data):
        if data[cursor:cursor + 4] != MAGIC:
            next_magic = data.find(MAGIC, cursor)
            end = len(data) if next_magic < 0 else next_magic
            if any(data[cursor:end]):
                if not interleaved:
                    raise ValueError("unexplained nonzero CUDA section bytes")
                state.setdefault("vendor_data_spans", []).append(
                    {"offset": cursor, "bytes": end - cursor, "sha256": sha(data[cursor:end])})
            cursor = end
            if cursor == len(data):
                break
        if cursor + 16 > len(data):
            raise ValueError("CUDA container header bounds")
        _, version, header_size, payload_size = struct.unpack_from("<IHHQ", data, cursor)
        end = cursor + header_size + payload_size
        if version != 1 or header_size < 16 or end > len(data):
            raise ValueError("CUDA container payload bounds")
        container = {"offset": cursor, "bytes": end - cursor,
                     "sha256": sha(data[cursor:end]), "entries": []}
        position = cursor + header_size
        while position < end:
            if position + 16 > end:
                raise ValueError("CUDA entry header bounds")
            kind, entry_version, header_bytes, size = struct.unpack_from("<HHIQ", data, position)
            if header_bytes < 16 or position + header_bytes + size > end:
                raise ValueError("CUDA entry payload bounds")
            payload_at = position + header_bytes
            payload = data[payload_at:payload_at + size]
            decoded, compression = decompress(payload)
            state["decoded_bytes"] += len(decoded)
            state["entries"] += 1
            if state["decoded_bytes"] > MAX_TOTAL or state["entries"] > 100000:
                raise ValueError("CUDA total decode limit")
            label = "cuda-" + str(state["entries"])
            emit(decoded, label)
            row = {"kind": kind, "version": entry_version, "offset": payload_at, "bytes": size,
                   "sha256": sha(payload), "compression": compression,
                   "decoded_bytes": len(decoded), "decoded_sha256": sha(decoded)}
            if kind == 2:
                row["elf"] = elf(decoded, lambda value, part: emit(value, label + "-" + part))
            elif kind == 1:
                if b".version" not in decoded[:8192] or b".target" not in decoded[:8192]:
                    raise ValueError("unrecognized CUDA PTX image")
                row["ptx"] = True
            else:
                raise ValueError("unrecognized CUDA entry kind")
            container["entries"].append(row)
            position += header_bytes + size
        if position != end:
            raise ValueError("CUDA container does not close exactly")
        found += 1
        yield container
        cursor = end
    if not found:
        raise ValueError("no CUDA containers")


def containers(data, emit, state=None, interleaved=False):
    return list(iter_containers(data, emit, state, interleaved))


def zstd_frames(data, emit, state):
    position = 0
    while True:
        start = data.find(b"\x28\xb5\x2f\xfd", position)
        if start < 0:
            break
        decoder = zstd.ZstdDecompressor()
        cursor, chunks, total = start, [], 0
        while not decoder.eof:
            block = data[cursor:cursor + (64 << 10)]
            if not block:
                raise ValueError("truncated embedded Zstandard frame")
            cursor += len(block)
            decoded = decoder.decompress(block, max_length=MAX_IMAGE - total + 1)
            chunks.append(decoded)
            total += len(decoded)
            if total > MAX_IMAGE or (not decoder.needs_input and not decoder.eof):
                raise ValueError("embedded Zstandard frame exceeds inspection bound")
        end = cursor - len(decoder.unused_data)
        decoded = b"".join(chunks)
        state["decoded_bytes"] += len(decoded)
        if state["decoded_bytes"] > MAX_TOTAL:
            raise ValueError("total Zstandard inspection bound")
        emit(decoded, "embedded-zstd-" + str(start))
        row = {"offset": start, "bytes": end - start, "sha256": sha(data[start:end]),
               "decoded_bytes": len(decoded), "decoded_sha256": sha(decoded)}
        if decoded.startswith(b"\x7fELF"):
            row["elf"] = elf(decoded, lambda value, name: emit(value, "embedded-" + name))
        yield row
        position = end
