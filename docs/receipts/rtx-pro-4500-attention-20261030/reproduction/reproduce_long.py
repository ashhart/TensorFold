"""Replay a public long-context code fixture against an already running native server."""
import argparse
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('base')
p.add_argument('fixture', type=Path)
p.add_argument('--model', default='bench')
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
body = json.loads(a.fixture.read_text())
body['model'] = a.model
samples = []
for drafted in [True, False]:
    for repeat in range(a.repetitions):
        request = {**body, 'draft': drafted}
        q = urllib.request.Request(a.base.rstrip('/') + '/v1/chat/completions',
                                   json.dumps(request).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(q, timeout=600) as response:
            result = json.load(response)
        samples.append({'drafted': drafted, 'repeat': repeat, 'request': request, 'response': result})
        a.output.write_text(json.dumps({'samples': samples}, indent=2) + '\n')
for drafted in [True, False]:
    selected = [x['response'] for x in samples if x['drafted'] == drafted]
    hashes = {x['tensorfold']['token_sha256'] for x in selected}
    assert len(hashes) == 1, 'Repeated output tokens differ'
    print(json.dumps({'drafted': drafted,
                      'median_decode_tps': statistics.median(x['tensorfold']['tokens_per_second'] for x in selected),
                      'token_sha256': next(iter(hashes))}))
assert len({x['response']['tensorfold']['token_sha256'] for x in samples}) == 1, 'Drafted/plain mismatch'
