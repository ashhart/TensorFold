"""Check native chat retention against a cache-off reference using the existing HTTP benchmark clients."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import bench_cache


def qualified(rows, count, retained, edited):
    if any(row.get('tokens') != 8 or type(row.get('cached_tokens')) is not int for row in rows):
        return False
    if not bench_cache.qualify(rows, count, not retained or not edited, retained):
        return False
    hashes = [row.get('token_sha256') for row in rows]
    if not all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in hashes):
        return False
    if len(set(hashes)) != 1:
        return False
    return not retained or not edited or rows[0].get('cached_tokens', 0) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('model')
    parser.add_argument('fixtures')
    parser.add_argument('mode', choices=('fresh', 'retained'))
    parser.add_argument('--reference', help='A fresh-mode output; compare every retained reply by full token hash')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fixtures = json.loads(Path(args.fixtures).read_text())
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    report = {'schema_version': 1, 'mode': args.mode, 'reply_tokens': 8, 'temperature': 0, 'seed': 1234,
              'client': 'bench_concurrent.stream/together and bench_cache.receipt/qualify',
              'client_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': []}
    retained = args.mode == 'retained'
    for index, fixture in enumerate(fixtures):
        item = fixture['item']
        first = bench_cache.bench.stream(args.base, args.model, item, 8, 0, 1234)
        rows = [bench_cache.receipt(first, 'first')]
        if retained:
            repeated = bench_cache.bench.stream(args.base, args.model, item, 8, 0, 1234)
            rows.append(bench_cache.receipt(repeated, 'repeated', first))
            concurrent = bench_cache.bench.together(args.base, args.model, [(item, 1234)], 8, 0)[0]
            rows.append(bench_cache.receipt(concurrent, 'concurrent', first))
        digest = hashlib.sha256(json.dumps(fixture['token_ids']).encode()).hexdigest()
        passed = (len(fixture['token_ids']) == fixture['count'] and digest == fixture['token_ids_sha256']) and qualified(rows, fixture['count'], retained, index > 0)
        record = {'fixture': fixture['name'], 'fixture_sha256': digest, 'rows': rows, 'qualified': passed}
        if reference is not None:
            candidates = [row for row in reference['records'] if row['fixture'] == fixture['name']]
            record['equals_fresh'] = (reference['mode'] == 'fresh' and reference['qualified'] and len(candidates) == 1
                                      and candidates[0]['fixture_sha256'] == digest
                                      and all(row['token_sha256'] == candidates[0]['rows'][0]['token_sha256'] for row in rows))
            record['qualified'] &= record['equals_fresh']
        report['records'].append(record)
    report['qualified'] = bool(report['records']) and all(row['qualified'] for row in report['records'])
    encoded = json.dumps(report, indent=2) + '\n'
    Path(args.output).write_text(encoded)
    print(encoded, end='')
    return 0 if report['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
