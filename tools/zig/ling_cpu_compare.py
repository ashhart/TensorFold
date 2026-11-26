"""Alternate two Ling CPU decoder binaries and refuse changed generated token IDs."""
import argparse
import json
import re
import subprocess


def run(binary, model, prompt, count, threads):
    command = [binary, model, prompt, str(count), str(threads)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=True)
    ids = [int(x) for x in result.stdout.split('generated:', 1)[1].split('\ndecode', 1)[0].split()]
    match = re.search(r'decode .*?\(([0-9.]+) tokens/s\)', result.stdout)
    if not match or not ids:
        raise RuntimeError('decoder did not emit generated IDs and a decode rate')
    return {'command': command, 'ids': ids, 'decode_tps': float(match[1]),
            'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('model')
    parser.add_argument('prompt_ids', help='comma-separated prompt token IDs')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    if min(args.count, args.threads, args.reps) < 1:
        parser.error('count, threads and reps must be positive')
    expected = None
    for rep in range(args.reps):
        for label, binary in [('baseline', args.baseline), ('candidate', args.candidate)]:
            row = run(binary, args.model, args.prompt_ids, args.count, args.threads)
            if expected is None:
                expected = row['ids']
            row.update(rep=rep, label=label, ids_equal=row['ids'] == expected)
            print(json.dumps(row), flush=True)
            if not row['ids_equal']:
                raise SystemExit('generated token mismatch')


if __name__ == '__main__':
    main()
