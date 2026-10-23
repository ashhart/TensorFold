"""Compare controlled JSONL receipts; a speed result is invalid if tokens differ."""
import argparse
import json
import statistics

p = argparse.ArgumentParser()
p.add_argument('baseline')
p.add_argument('candidate')
a = p.parse_args()


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if 'request' not in r:
                continue
            rows[(r['cell'], r['rep'])] = r
    return rows


baseline, candidate = load(a.baseline), load(a.candidate)
groups = {}
for key, now in candidate.items():
    if key[0] == 'prefill-warmup':
        continue
    old = baseline[key]
    assert old['request'] == now['request'], (key, 'request differs')
    x, y = old['result'], now['result']
    assert x['tensorfold']['token_sha256'] == y['tensorfold']['token_sha256'], (key, 'tokens differ')
    assert x['usage']['prompt_tokens'] == y['usage']['prompt_tokens'], key
    groups.setdefault(key[0], []).append((old, now))

for cell, pairs in groups.items():
    def metrics(i):
        results = [p[i]['result'] for p in pairs]
        return {
            'wall_s': statistics.median(p[i]['wall'] for p in pairs),
            'decode_tps': statistics.median(r['tensorfold']['tokens_per_second'] for r in results),
            'prefill_s': statistics.median(r['tensorfold']['prefill_seconds'] for r in results),
            'prompt_tokens': results[0]['usage']['prompt_tokens'],
            'output_tokens': results[0]['usage']['completion_tokens'],
            'sha': results[0]['tensorfold']['token_sha256'],
        }
    print(json.dumps({'cell': cell, 'reps': len(pairs), 'token_equal': True,
                      'baseline': metrics(0), 'candidate': metrics(1)}, ensure_ascii=False))
