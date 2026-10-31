"""Extract one real CUDA fatbinary containing a required public symbol from a native ELF."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('symbol')
p.add_argument('output', type=Path)
a = p.parse_args()
binary = a.binary.read_bytes()
found = []
start = 0
while True:
    offset = binary.find(bytes.fromhex('50ed55ba'), start)
    if offset < 0:
        break
    start = offset + 4
    if offset + 16 > len(binary):
        continue
    magic, version, header, payload = struct.unpack_from('<IHHQ', binary, offset)
    if version != 1 or header != 16 or payload < 1024 or payload > len(binary) - offset - 16:
        continue
    image = binary[offset:offset + 16 + payload]
    if a.symbol.encode() in image:
        found.append((offset, image))
assert len(found) == 1, 'Expected one complete CUDA image containing the symbol'
offset, image = found[0]
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_bytes(image)
print(json.dumps({'binary_sha256': hashlib.sha256(binary).hexdigest(),
                  'fatbin_sha256': hashlib.sha256(image).hexdigest(),
                  'binary_offset': offset, 'image_bytes': len(image)}))
