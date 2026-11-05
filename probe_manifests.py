# probe_manifests.py LIB: which input files named in the restored build manifests can't be found on this runner?
import collections, os, re, sys
lib = sys.argv[1]
local = '.zig-cache'
roots = {0: '.', 1: lib, 2: local, 3: local, 4: '.'}
pat = re.compile(rb'([\x00-\x1f])([^\x00-\x1f]{3,400})\x00')
seen = collections.Counter(); missing = collections.Counter(); samples = {}; absolute = collections.Counter(); n = 0
for name in os.listdir(os.path.join(local, 'h')):
    p = os.path.join(local, 'h', name)
    if not os.path.isfile(p) or os.path.getsize(p) > 6000000: continue
    b = open(p, 'rb').read()
    if b'std/std.zig' not in b and b'std\\std.zig' not in b: continue
    n += 1
    for m in pat.finditer(b):
        pre, path = m[1][0] >> 2, m[2].decode(errors='replace')
        if pre not in roots or not re.search(r'[/\\.]', path): continue
        isabs = os.path.isabs(path) or re.match(r'^[A-Za-z]:[/\\]', path) is not None
        full = path if isabs else os.path.join(roots[pre], path)
        key = '/'.join(re.split(r'[/\\]', path)[:5]) if isabs else f'p{pre}:' + re.split(r'[/\\]', path)[0]
        seen[key] += 1
        if isabs: absolute[key] += 1
        if not os.path.exists(full):
            missing[key] += 1; samples.setdefault(key, (name[:12], path[:110]))
print('PROBE MF manifests', n, 'distinct roots', len(seen))
print('PROBE MF absolute', absolute.most_common(12))
print('PROBE MF missing', missing.most_common(14))
for k, v in list(samples.items())[:14]: print('PROBE MF sample', k, v)
