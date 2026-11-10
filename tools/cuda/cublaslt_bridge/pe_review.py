"""Bounded PE32+ structure review; this module never loads the inspected image."""
import hashlib
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def review(data):
    def unpack(fmt, at):
        if at < 0 or at + struct.calcsize(fmt) > len(data):
            raise ValueError("PE structure exceeds file")
        return struct.unpack_from(fmt, data, at)

    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("missing DOS header")
    peoff, = unpack("<I", 60)
    if peoff > 1 << 20 or data[peoff:peoff + 4] != b"PE\0\0":
        raise ValueError("invalid PE signature")
    machine, count, timestamp, symbols, nsyms, optional_size, flags = unpack("<HHIIIHH", peoff + 4)
    optional = peoff + 24
    magic, = unpack("<H", optional)
    if machine != 0x8664 or magic != 0x20B or not 1 <= count <= 96 or optional_size < 112:
        raise ValueError("expected bounded AMD64 PE32+")
    directory_count, = unpack("<I", optional + 108)
    if directory_count > 16 or optional_size < 112 + directory_count * 8:
        raise ValueError("invalid PE data directories")
    directories = [unpack("<II", optional + 112 + i * 8) for i in range(directory_count)]
    directories += [(0, 0)] * (16 - len(directories))
    sections = []
    for index in range(count):
        name, virtual_size, rva, size, at, _, _, _, _, attributes = unpack(
            "<8sIIIIIIHHI", optional + optional_size + index * 40)
        if at + size > len(data):
            raise ValueError("PE section exceeds file")
        sections.append({"name": name.rstrip(b"\0").decode("ascii", "strict"),
                         "rva": rva, "virtual_size": virtual_size, "offset": at, "bytes": size,
                         "flags": attributes, "sha256": sha(data[at:at + size])})
    ranges = sorted((row["offset"], row["offset"] + row["bytes"]) for row in sections if row["bytes"])
    if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("overlapping PE raw sections")

    def offset(rva, size=1):
        for row in sections:
            delta = rva - row["rva"]
            if 0 <= delta and delta + size <= row["bytes"]:
                return row["offset"] + delta
        raise ValueError("PE RVA exceeds file-backed section")

    def string(rva):
        at = offset(rva)
        end = data.find(b"\0", at, min(at + 65536, len(data)))
        if end < 0:
            raise ValueError("unterminated PE string")
        return data[at:end].decode("ascii", "strict")

    imports = []
    address, length = directories[1]
    if address:
        at = offset(address, length)
        for index in range(min(length // 20, 4096)):
            original, _, _, name, first = unpack("<IIIII", at + index * 20)
            if not any(unpack("<IIIII", at + index * 20)):
                break
            functions, thunk = [], offset(original or first, 8)
            for slot in range(65536):
                value, = unpack("<Q", thunk + slot * 8)
                if not value:
                    break
                functions.append({"ordinal": value & 65535} if value >> 63 else {"name": string(value + 2)})
            else:
                raise ValueError("PE import thunk limit")
            imports.append({"dll": string(name), "functions": functions})
        else:
            raise ValueError("PE import table missing terminator")
    if directories[13] != (0, 0):
        raise ValueError("delayed imports require an explicit reviewed parser")

    exports = []
    address, length = directories[0]
    if address:
        at = offset(address, 40)
        fields = unpack("<IIHHIIIIIII", at)
        functions, names, names_at = fields[6], fields[7], fields[9]
        if functions > 200000 or names > functions:
            raise ValueError("PE export count limit")
        table = offset(names_at, names * 4)
        exports = [string(unpack("<I", table + i * 4)[0]) for i in range(names)]
        if len(set(exports)) != len(exports):
            raise ValueError("duplicate PE named exports")
    debug = []
    address, length = directories[6]
    if address:
        if length % 28 or length > 28 * 4096:
            raise ValueError("PE debug size limit")
        at = offset(address, length)
        for index in range(length // 28):
            _, _, _, _, kind, size, _, raw = unpack("<IIHHIIII", at + index * 28)
            if raw + size > len(data):
                raise ValueError("PE debug payload exceeds file")
            payload = data[raw:raw + size]
            row = {"type": kind, "offset": raw, "bytes": size, "sha256": sha(payload)}
            if kind == 2:
                if payload[:4] != b"RSDS" or size < 25:
                    raise ValueError("unknown CodeView record")
                pdb = payload[24:].split(b"\0", 1)[0]
                row.update(pdb_sha256=sha(pdb), pdb_has_path=b"/" in pdb or b"\\" in pdb)
            debug.append(row)
    resources = []
    address, length = directories[2]
    if address:
        base = offset(address, length)
        visited = set()

        def walk(relative, route):
            if relative in visited or len(visited) > 4096 or len(route) > 3:
                raise ValueError("PE resource recursion limit")
            visited.add(relative)
            if relative + 16 > length:
                raise ValueError("resource directory bounds")
            _, _, _, _, named, numbered = unpack("<IIHHHH", base + relative)
            if relative + 16 + 8 * (named + numbered) > length:
                raise ValueError("resource entry bounds")
            for index in range(named + numbered):
                name, value = unpack("<II", base + relative + 16 + 8 * index)
                if name >> 31:
                    position = name & 0x7FFFFFFF
                    if position + 2 > length:
                        raise ValueError("resource name bounds")
                    chars, = unpack("<H", base + position)
                    if position + 2 + chars * 2 > length:
                        raise ValueError("resource name text bounds")
                    key = data[base + position + 2:base + position + 2 + chars * 2].decode("utf-16le")
                else:
                    key = name
                if value >> 31:
                    walk(value & 0x7FFFFFFF, route + [key])
                else:
                    if value + 16 > length:
                        raise ValueError("resource leaf bounds")
                    rva, size, codepage, reserved = unpack("<IIII", base + value)
                    begin = offset(rva, size)
                    resources.append({"route": route + [key], "offset": begin, "bytes": size,
                                      "codepage": codepage, "reserved": reserved,
                                      "sha256": sha(data[begin:begin + size])})
        walk(0, [])
    certificate = []
    cert_at, cert_size = directories[4]
    if cert_at or cert_size:
        if cert_at % 8 or cert_at + cert_size != len(data) or cert_size > 4 << 20:
            raise ValueError("PE certificate bounds or unexplained certificate overlay")
        cursor = cert_at
        while cursor < cert_at + cert_size:
            size, revision, kind = unpack("<IHH", cursor)
            if size < 8 or cursor + size > cert_at + cert_size or revision != 0x200 or kind != 2:
                raise ValueError("unsupported WIN_CERTIFICATE")
            certificate.append({"offset": cursor + 8, "bytes": size - 8,
                                "sha256": sha(data[cursor + 8:cursor + size])})
            cursor += (size + 7) & ~7
        if cursor != cert_at + cert_size:
            raise ValueError("PE certificate alignment mismatch")
    last = max(end for _, end in ranges)
    overlay_end = cert_at if cert_size else len(data)
    if overlay_end < last:
        raise ValueError("certificate overlaps PE section")
    overlay = data[last:overlay_end]
    if any(overlay):
        raise ValueError("nonzero unexplained PE overlay")
    if symbols or nsyms:
        raise ValueError("COFF symbols must be stripped from the downloadable image")
    return {"machine": machine, "dll": bool(flags & 0x2000), "timestamp": timestamp,
            "image_base": unpack("<Q", optional + 24)[0], "sections": sections, "imports": imports, "exports": exports, "debug": debug,
            "resources": resources, "certificate": certificate,
            "overlay_zero_bytes": len(overlay), "COFF_symbols": nsyms}


def cuda_wrappers(data, pe):
    result = []
    for section in pe["sections"]:
        if section["name"] != ".nvFatBi":
            continue
        size = section["virtual_size"]
        if size > section["bytes"] or size % 24:
            raise ValueError("CUDA wrapper table geometry")
        for position in range(0, size, 24):
            at = section["offset"] + position
            magic, version, target, linked = struct.unpack_from("<IIQQ", data, at)
            if magic != 0x466243B1 or version not in (1, 2) or not target or (version == 1 and linked):
                raise ValueError("unrecognized CUDA registration wrapper")
            targets = []
            for pointer in [target]:
                rva = pointer - pe["image_base"]
                match = next((row for row in pe["sections"] if row["rva"] <= rva
                              < row["rva"] + row["bytes"]), None)
                if match is None or match["name"] not in (".nv_fatb", ".cask_re"):
                    raise ValueError("CUDA wrapper points outside reviewed payload sections")
                raw = match["offset"] + rva - match["rva"]
                if data[raw:raw + 4] != b"\x50\xed\x55\xba":
                    raise ValueError("CUDA wrapper does not point to a container")
                targets.append({"section": match["name"], "offset": raw})
            prelinked = []
            if version == 2 and linked:
                rva = linked - pe["image_base"]
                table = next((row for row in pe["sections"] if row["name"] in (".data", ".rdata")
                              and row["rva"] <= rva < row["rva"] + row["bytes"]), None)
                if table is None:
                    raise ValueError("prelinked CUDA fatbin array is outside reviewed host data")
                cursor = table["offset"] + rva - table["rva"]
                for _ in range(4096):
                    if cursor + 8 > table["offset"] + table["bytes"]:
                        raise ValueError("unterminated prelinked CUDA fatbin array")
                    value, = struct.unpack_from("<Q", data, cursor)
                    cursor += 8
                    if not value:
                        break
                    item_rva = value - pe["image_base"]
                    linked_section = next((row for row in pe["sections"]
                                           if row["rva"] <= item_rva < row["rva"] + row["bytes"]), None)
                    if linked_section is None or linked_section["name"] not in (".nv_fatb", ".nvFatBi", ".cask_re"):
                        raise ValueError("prelinked CUDA array points outside reviewed payloads")
                    prelinked.append(item_rva)
                else:
                    raise ValueError("prelinked CUDA fatbin array limit")
            result.append({"offset": at, "version": version, "targets": targets,
                           "prelinked_array_rvas": prelinked})
        if any(data[section["offset"] + size:section["offset"] + section["bytes"]]):
            raise ValueError("unexplained CUDA wrapper-table padding")
    return result
