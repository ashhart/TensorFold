"""Launch/client wrapper: one cold request per fresh pair, with fail-closed receipt comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


def require(ok, why):
    if not ok:
        raise ValueError(why)


def launch(args):
    env = os.environ.copy()
    env.pop('TF_GLM_PREFILL_TRACE', None)
    env['TF_GLM_PROF'] = '0'
    env.pop('TF_GLM_LAYERS', None)
    if args.prefix:
        require(not any(ord(c) < 32 or ord(c) == 127 for c in args.prefix), 'control character in prefix')
        env['TF_GLM_PREFILL_TRACE'] = args.prefix
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    cmd = [args.binary, 'serve', args.checkpoint, '--tp', '2', '--rank', str(args.rank),
           '--master', args.master, '--master-port', str(args.master_port), '--port', str(args.port),
           '--context', '33792', '--parallel', '1', '--prompt-cache-gib', '0', *extra]
    if args.dry_run:
        print(shlex.join(['env', '-u', 'TF_GLM_PREFILL_TRACE', '-u', 'TF_GLM_LAYERS', 'TF_GLM_PROF=0',
                          *(['TF_GLM_PREFILL_TRACE=' + args.prefix] if args.prefix else []), *cmd]))
    else:
        os.execvpe(cmd[0], cmd, env)


def measure(args):
    # Tokenization does not warm the model; send one inference request after both ranks are ready.
    import urllib.request
    sys.path.insert(0, str(Path(args.repo).resolve() / 'tools'))
    import bench_openai
    corpus = Path(args.corpus).read_text()
    req = urllib.request.Request(args.base + '/tokenize', json.dumps({
        'prompt': corpus, 'add_special_tokens': False}).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        ids = json.load(response)['tokens']
    require(len(ids) >= args.tokens, 'corpus is too short')
    ids = ids[:args.tokens]
    result = bench_openai.stream(args.base, args.model,
                                {'name': 'cold', 'kind': 'completion', 'prompt': ids}, 64, 0, 1234)
    require(result.get('complete') and result.get('cached_tokens') == 0, 'request failed or reused a prefix')
    require(result.get('prompt_tokens') == args.tokens, 'server prompt length differs')
    require(result.get('tokens') == 64 and re.fullmatch('[0-9a-f]{64}', result.get('token_sha256') or ''),
            'missing complete token-id hash')
    result.pop('text', None)
    result.pop('pieces', None)
    result['input_ids_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


def partition(view, rows, classes):
    require(isinstance(view, dict), 'missing additive partition')
    total = view.get('total_ns')
    require(type(total) is int and total >= 0 and view.get('accounted_ns') == total, 'nonconserving partition')
    exclusive = view.get('exclusive_by_row_ns')
    require(isinstance(exclusive, list) and len(exclusive) == len(rows), 'wrong exclusive row count')
    sums = [0] * len(rows)
    buckets = dict.fromkeys(('stage', 'parent_self', 'overlap', 'gap'), 0)
    cursor = 0
    for seg in view.get('segments', []):
        end = seg.get('end_ns')
        require(seg.get('begin_ns') == cursor and type(end) is int and cursor < end <= total,
                'partition gap, overlap or invalid endpoint')
        bucket, row = seg.get('bucket'), seg.get('row')
        require(bucket in buckets, 'unknown partition bucket')
        buckets[bucket] += end - cursor
        if bucket in ('stage', 'parent_self'):
            require(type(row) is int and 0 <= row < len(rows), 'invalid selected row')
            sums[row] += end - cursor
        else:
            require(row is None, 'overlap/gap must not invent a selected row')
        cursor = end
    require(cursor == total and sums == exclusive, 'partition does not cover request or match row totals')
    for bucket in ('parent_self', 'overlap', 'gap'):
        require(view.get(bucket + '_ns') == buckets[bucket], 'wrong bucket total: ' + bucket)
    require(buckets['gap'] == 0, 'unaccounted request interval')
    class_ns = [0] * len(classes)
    for row, ns in zip(rows, exclusive):
        require(row.get('class') in classes, 'unknown stage class')
        class_ns[classes.index(row['class'])] += ns
    require(view.get('class_ns') == class_ns and sum(class_ns) + buckets['overlap'] == total,
            'class totals double count or omit time')


def read_ledgers(args):
    ranks, receipts = set(), []
    for path in args.ledgers:
        data = json.loads(Path(path).read_text())
        require(data.get('schema') == 'tf-glm-prefill-v2' and data.get('world') == 2, 'wrong ledger schema/world')
        require(data.get('complete') is True and data.get('dropped_stages') == 0, 'incomplete ledger')
        rows = data.get('rows')
        require(data.get('prompt_tokens') == args.tokens and isinstance(rows, list) and rows, 'wrong or empty ledger')
        require(data['rank'] in (0, 1) and data['rank'] not in ranks, 'wrong or duplicate rank')
        ranks.add(data['rank'])
        for index, row in enumerate(rows):
            require(row.get('sequence') == index and row.get('event_status') == 'complete', 'incomplete row')
            parent = row.get('parent')
            require((index == 0 and parent is None and row.get('stage') == 'request') or
                    (index > 0 and type(parent) is int and 0 <= parent < index), 'invalid parent')
            for name in ('cuda_event_ns', 'cuda_begin_ns', 'cuda_end_ns', 'wall_begin_ns', 'wall_end_ns'):
                require(type(row.get(name)) is int and row[name] >= 0, 'unavailable timing: ' + name)
            require(row['cuda_end_ns'] >= row['cuda_begin_ns'] and row['wall_end_ns'] >= row['wall_begin_ns'],
                    'backwards interval')
        stages = {row['stage'] for row in rows}
        required = {'request', 'chunk', 'head', 'experts', 'dense_gemm', 'norms', 'all_gather'}
        if getattr(args, 'paired', False):
            required |= {'paired_prefill', 'ffn', 'shared_expert', 'output_projection', 'commit'}
        if getattr(args, 'mtp', False):
            required |= {'mtp', 'mtp_layer', 'mtp_projection'}
        if getattr(args, 'joint', False):
            required |= {'joint_ffn'}
        if getattr(args, 'images', False):
            required |= {'image_splice'}
        require(required <= stages, 'missing required GLM stages: ' + ','.join(sorted(required - stages)))
        classes = data.get('class_order')
        require(isinstance(classes, list) and classes and len(set(classes)) == len(classes), 'invalid class order')
        for axis in ('gpu', 'wall'):
            view = data.get(axis + '_accounting')
            partition(view, rows, classes)
            first = 'cuda' if axis == 'gpu' else 'wall'
            require(view['total_ns'] == rows[0][first + '_end_ns'] - rows[0][first + '_begin_ns'],
                    'partition differs from whole request interval')
        receipts.append((Path(path), data))
    require(ranks == {0, 1}, 'both rank ledgers are required')
    return receipts


def collect(args):
    receipts = read_ledgers(args)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    for path, data in receipts:
        (target / ('prefill.rank%d.json' % data['rank'])).write_bytes(path.read_bytes())
    print('PASS: collected both complete rank receipts with conserved GPU and wall timelines')


def verify(args):
    off, on = (json.loads(Path(p).read_text()) for p in (args.off, args.on))
    for receipt in (off, on):
        require(receipt.get('complete') and receipt.get('cached_tokens') == 0, 'not a complete cold receipt')
        require(receipt.get('prompt_tokens') == args.tokens and receipt.get('tokens') == 64, 'wrong token count')
        require(re.fullmatch('[0-9a-f]{64}', receipt.get('token_sha256') or ''), 'missing token-id hash')
    for field in ('input_ids_sha256', 'token_sha256', 'prompt_tokens', 'cached_tokens', 'tokens'):
        require(off.get(field) is not None and off[field] == on.get(field), 'mismatch: ' + field)
    for name in ('rounds', 'drafted', 'accepted'):
        left, right = off.get('speculative', {}).get(name), on.get('speculative', {}).get(name)
        require(type(left) is int and type(right) is int and left == right, 'missing/mismatched counter: ' + name)
    read_ledgers(args)
    print('PASS: both complete rank ledgers; exact input/output token hashes, cold counts and draft counters')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    p = modes.add_parser('launch')
    p.add_argument('--binary', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rank', required=True, type=int, choices=(0, 1))
    p.add_argument('--master', required=True)
    p.add_argument('--master-port', type=int, default=29583)
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--prefix')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER)
    p = modes.add_parser('measure')
    p.add_argument('--repo', default='.')
    p.add_argument('--base', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--tokens', required=True, type=int, choices=(8192, 32768))
    p.add_argument('--output', required=True)
    for name in ('collect', 'verify'):
        p = modes.add_parser(name)
        p.add_argument('--tokens', required=True, type=int, choices=(8192, 32768))
        for flag in ('paired', 'mtp', 'joint', 'images'):
            p.add_argument('--' + flag, action='store_true')
        p.add_argument('ledgers', nargs=2)
    modes.choices['collect'].add_argument('--output', required=True)
    p = modes.choices['verify']
    p.add_argument('--off', required=True)
    p.add_argument('--on', required=True)
    args = parser.parse_args()
    try:
        globals()[args.mode](args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
