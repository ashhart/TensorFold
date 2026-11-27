#!/usr/bin/env python3
"""Lead-run cold CUDA prompt pairs; each arm owns one process and sends one inference request."""
import argparse, hashlib, json, os, re, socket, subprocess, time, urllib.request
from pathlib import Path
SCHEMA = 'tensorfold.cuda-prefill-stage.v1'
def digest(data): return hashlib.sha256(data).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')
def request(base, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as reply: return json.load(reply)
def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def whole(value):
    return type(value) is int and value >= 0

def fresh(row):
    state = row.get('restoration')
    require(isinstance(state, dict), 'missing restoration fields')
    require(state.get('kind') == 'fresh' and type(state.get('tokens_restored')) is int
            and state['tokens_restored'] == 0 and state.get('kv_restored') is False
            and state.get('recurrent_restored') is False and state.get('restore_failed') is False,
            'cold request reports restored or failed state')

def records(text):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            require(SCHEMA not in line, f'malformed ledger line {number}')
            continue
        if isinstance(row, dict) and row.get('schema') == SCHEMA:
            rows.append(row)
    return rows

def ledger(text, n, drafted):
    rows = records(text)
    require(rows, 'trace-on emitted no ledger')
    seen = set()
    for row in rows:
        sequence = row.get('sequence')
        require(whole(sequence) and sequence not in seen, 'invalid or duplicate sequence')
        seen.add(sequence)
        require(row.get('complete') is True and type(row.get('dropped_stages')) is int
                and row['dropped_stages'] == 0, 'incomplete stage')
        status = row.get('event_status')
        require(status in ('not_launched', 'complete', 'capture_only'), 'unavailable or pending event time')
        if status == 'complete':
            require(whole(row.get('cuda_event_ns')), 'missing CUDA event time')
        memory = row.get('memory')
        require(isinstance(memory, dict), 'missing physical memory fields')
        require(whole(memory.get('rss_bytes')), 'missing physical RSS')
        require(whole(memory.get('physical_peak_bytes')), 'missing physical peak')
    restores = [row for row in rows if row.get('stage') == 'restore']
    require(len(restores) == 1, 'expected one restore status record')
    fresh(restores[0])
    chunks = sorted((row for row in rows if row.get('stage') == 'target_chunk'), key=lambda row: row['sequence'])
    at = 0
    for row in chunks:
        require(row.get('start_token') == at and whole(row.get('rows')) and row['rows'] > 0,
                'target chunks do not cover the prompt in order')
        at += row['rows']
        require(row.get('shape_bucket_rows') == row['rows'] and row.get('shape_bucket_segments') == 1,
                'wrong prompt shape bucket')
        require(row.get('graph_mode') == 'eager', 'target prompt was not eager')
        require(row.get('event_status') == 'complete' and whole(row.get('cuda_event_ns')), 'target event time unavailable')
        fresh(row)
    require(at == n, 'wrong total prompt length')
    tails = [row for row in rows if row.get('stage') == 'trailing_absorb']
    require(len(tails) == int(drafted), 'missing or duplicate final-row absorb')
    for row in tails:
        require(row.get('start_token') == n - 1 and row.get('rows') == 1,
                'trailing absorb is not the real final prompt row')
    for row in rows:
        if row.get('stage') == 'drafter_absorb':
            require(whole(row.get('start_token')) and whole(row.get('rows'))
                    and row['start_token'] + row['rows'] <= n - 1, 'ordinary absorb overlaps the final prompt row')
    return rows

def drafter_evidence(text, drafted, traced, rows, counters, expected='DFlash2'):
    labels = re.findall(r'drafts: ([^\n]*?); context:', text)
    require(labels, 'missing server drafter identity')
    sources = labels[-1].strip()
    names = re.sub(r'\([^)]*\)', '', sources).split('+')
    def normal(value):
        return ''.join(c for c in value.casefold() if c.isalnum())
    available = [normal(name) for name in names]
    proposed = counters.get('drafted')
    accepted = counters.get('accepted')
    require(whole(proposed) and whole(accepted), 'missing proposal counters')
    opened = [row for row in rows if row.get('stage') == 'drafter_open']
    tails = [row for row in rows if row.get('stage') == 'trailing_absorb']
    if drafted:
        require(normal(expected) in available, f'drafted arm did not load {expected}: {sources}')
        require(proposed > 0, 'drafted arm proposed no tokens')
        if traced:
            require(opened and tails, 'drafted arm emitted no prompt drafter work')
    else:
        require(normal(sources) in ('copiesonly', 'none', 'off'), f'plain arm loaded a drafter: {sources}')
        require(proposed == 0 and accepted == 0, 'plain arm reported drafted tokens')
        if traced:
            require(not opened and not tails, 'plain arm emitted drafter prompt work')
    return dict(request_draft=drafted, expected_drafter=expected if drafted else None,
                observed_sources=sources, proposed_tokens=proposed, accepted_tokens=accepted,
                traced_prompt_activity=bool(opened and tails) if traced else None,
                validated=True)

def sample_sequence(drafted=True):
    def row(stage, sequence, **extra):
        return dict(schema=SCHEMA, stage=stage, sequence=sequence, complete=True, dropped_stages=0,
                    event_status='not_launched', cuda_event_ns=None, graph_mode='eager',
                    restoration=dict(kind='fresh', tokens_restored=0, kv_restored=False,
                                     recurrent_restored=False, restore_failed=False),
                    memory=dict(rss_bytes=1024, physical_peak_bytes=2048), **extra)
    rows = [row('restore', 0), row('target_chunk', 1, start_token=0, rows=4096,
                                 shape_bucket_rows=4096, shape_bucket_segments=1)]
    rows[1].update(event_status='complete', cuda_event_ns=1000)
    if drafted:
        rows += [row('drafter_absorb', 2, start_token=0, rows=4095),
                 row('trailing_absorb', 3, start_token=4095, rows=1)]
    return rows

def controls():
    import copy
    def text(rows):
        return '\n'.join(map(json.dumps, rows))
    valid = sample_sequence()
    for drafted in (False, True):
        ledger(text(sample_sequence(drafted)), 4096, drafted)
    bad = {}
    def case(name):
        rows = copy.deepcopy(valid)
        bad[name] = rows
        return rows
    case('incomplete')[1]['complete'] = False
    bad['missing_tail'] = copy.deepcopy(valid[:-1])
    bad['false_tail'] = copy.deepcopy(valid[:-1])
    bad['false_tail'][-1]['stage'] = 'trailing_absorb'
    case('wrong_length')[1].update(rows=4095, shape_bucket_rows=4095)
    case('restored_restore')[0]['restoration'].update(kind='snapshot', tokens_restored=1,
                                                     kv_restored=True, recurrent_restored=True)
    case('restored_chunk')[1]['restoration'].update(kind='snapshot', tokens_restored=1)
    case('missing_rss')[1]['memory']['rss_bytes'] = None
    case('missing_peak')[1]['memory']['physical_peak_bytes'] = None
    case('unavailable_time')[1].update(event_status='unavailable', cuda_event_ns=None)
    case('pending_time')[1].update(event_status='pending', cuda_event_ns=None)
    results = []
    for name, rows in bad.items():
        try:
            ledger(text(rows), 4096, True)
        except ValueError as error:
            results.append(dict(name=name, outcome='rejected', reason=str(error), input_json=json.dumps(rows)))
        else:
            raise RuntimeError(f'planted control unexpectedly accepted: {name}')
    draft_rows = copy.deepcopy(valid)
    opened = copy.deepcopy(valid[2]); opened.update(stage='drafter_open', rows=0, start_token=0)
    draft_rows.insert(2, opened)
    for index, row in enumerate(draft_rows): row['sequence'] = index
    header = '[tensorfold] serving test (drafts: DFlash2 (example/drafter) + copies; context: 4352)'
    counts = dict(drafted=8, accepted=4)
    drafter_evidence(header, True, True, draft_rows, counts)
    drafter_evidence('serving (drafts: copies only; context: 4352)', False, True, [], dict(drafted=0, accepted=0))
    draft_bad = []
    for name, line, frames, proposals in [
        ('copies_only_in_drafted_arm', 'serving (drafts: copies only; context: 4352)', draft_rows, counts),
        ('no_proposals', header, draft_rows, dict(drafted=0, accepted=0)),
        ('no_prompt_drafter_work', header, sample_sequence(False), counts),
    ]:
        try:
            drafter_evidence(line, True, True, frames, proposals)
        except ValueError as error:
            draft_bad.append(dict(name=name, outcome='rejected', reason=str(error)))
        else:
            raise RuntimeError(f'planted drafter control unexpectedly accepted: {name}')
    return dict(schema='cuda-prefill-parser-controls.v1' , network_or_device_run=False,
                source_sha256=digest(Path(__file__).read_bytes()), positives=['plain', 'drafted_final_row'], controls=results, drafter_controls=draft_bad)

def run(args, n, drafted, enabled, repetition):
    tag = f'{n}-draft{int(drafted)}-trace{enabled}-rep{repetition}'
    dest = args.output / tag; dest.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    cmd = [str(args.binary.resolve()), 'serve', str(args.model_dir.resolve()), '--backend','cuda',
           '--host','127.0.0.1','--port',str(port),'--name','e440','--no-update-check','--no-thinking',
           '--context',str(max(args.lengths) + 256),'--parallel','1','--prompt-cache-gib','0',
           '--drafter',str(args.drafter.resolve()) if drafted else 'none', *args.server_arg]
    env = {**os.environ, 'TF_CUDA_PREFILL_TRACE':str(enabled)}
    log = dest / 'server.log'
    with log.open('wb') as out:
        process = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.start_timeout
            while True:
                if process.poll() is not None: raise RuntimeError(f'server exited {process.returncode}: {log}')
                try: request(base, '/health'); break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline: raise TimeoutError('server readiness')
                    time.sleep(.25)
            prompt_file = args.output / f'prompt-{n}.json'
            if not prompt_file.exists() and args.prompts_from:
                prompt_file.write_bytes((args.prompts_from / prompt_file.name).read_bytes())
            if not prompt_file.exists():
                unit = request(base, '/v1/tokenize', {'prompt':'Explain the following program and its memory use.\nfor i in range(100):\n    print(i * i)\n', 'add_special_tokens':False})['tokens']
                assert unit and all(type(x) is int and x >= 0 for x in unit)
                write(prompt_file, (unit * ((n + len(unit) - 1)//len(unit)))[:n])
            prompt = json.loads(prompt_file.read_text()); assert len(prompt) == n
            body = {'model':'e440','prompt':prompt,'max_tokens':64,'temperature':0,'seed':7,
                    'draft':drafted,'ignore_eos':True,'stream':False}
            write(dest/'request.json',body)
            began = time.perf_counter(); reply = request(base, '/v1/completions', body)
            elapsed = time.perf_counter() - began
            write(dest/'reply.json',reply)
            assert 'error' not in reply
            usage = reply['usage']; fingerprint = reply['tensorfold']['token_sha256']
            assert re.fullmatch('[0-9a-f]{64}',fingerprint)
            assert usage['prompt_tokens'] == n and usage['completion_tokens'] == 64
            assert usage['prompt_tokens_details']['cached_tokens'] == 0
            result = {'prompt_tokens':n,'prompt_sha256':digest(prompt_file.read_bytes()),'token_sha256':fingerprint,
                      'completion_tokens':64,'cached_tokens':0,'wall_seconds':elapsed,'runtime':reply['tensorfold'],
                      'speculative':{k:(reply.get('speculative') or {}).get(k) for k in ('rounds','drafted','accepted')},'command':cmd,'trace':enabled,'draft':drafted,
                      'binary_sha256':digest(args.binary.read_bytes())}
            write(dest/'receipt.json',result)
        finally:
            process.terminate()
            try: process.wait(timeout=120)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    text = log.read_text(errors='replace')
    try:
        observed = drafter_evidence(text, drafted, bool(enabled), records(text), result['speculative'], args.expected_drafter)
    except ValueError as error:
        write(dest/'drafter-evidence.json', dict(request_draft=drafted, validated=False, problem=str(error)))
        raise
    write(dest/'drafter-evidence.json', observed)
    if enabled: write(dest/'ledger.json',ledger(text,n,drafted))
    else: assert SCHEMA not in text, 'trace-off emitted a ledger'
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path); p.add_argument('--model-dir',type=Path)
    p.add_argument('--drafter',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--lengths',type=lambda x:list(map(int,x.split(','))),default=[4096,32768,132096])
    p.add_argument('--draft-arms',type=lambda x:list(map(int,x.split(','))),default=[0,1],help='0: plain, 1: drafter; select 1 to validate the drafted arm first')
    p.add_argument('--expected-drafter',default='DFlash2')
    p.add_argument('--trace-arms',type=lambda x:list(map(int,x.split(','))),default=[0,1]); p.add_argument('--prompts-from',type=Path)
    p.add_argument('--reps',type=int,default=3); p.add_argument('--start-timeout',type=float,default=1800)
    p.add_argument('--server-arg',action='append',default=[])
    p.add_argument('--self-test', action='store_true', help='offline parser controls; never starts a server')
    p.add_argument('--controls-output', type=Path, help='write the reproducible offline control inputs and failures')
    args = p.parse_args()
    if args.self_test:
        result = controls()
        if args.controls_output:
            args.controls_output.parent.mkdir(parents=True, exist_ok=True)
            write(args.controls_output, result)
        print(f"PASS: {len(result['controls'])} parser and {len(result['drafter_controls'])} drafter controls rejected; no network or device run")
        return
    if not all((args.binary, args.model_dir, args.drafter, args.output)):
        p.error('device run requires --binary, --model-dir, --drafter and --output')
    if args.controls_output:
        p.error('--controls-output requires --self-test')
    args.output.mkdir(parents=True,exist_ok=True)
    assert not any(args.output.iterdir()), 'use a new evidence directory'
    assert args.trace_arms and set(args.trace_arms) <= {0,1}
    assert args.draft_arms and set(args.draft_arms) <= {0,1}
    pairs = []
    for n in args.lengths:
        for mode in args.draft_arms:
            drafted = bool(mode)
            for rep in range(args.reps):
                order = (0,1) if rep%2 == 0 else (1,0)
                arms = {arm:run(args,n,drafted,arm,rep) for arm in order if arm in args.trace_arms}
                for key in (() if set(arms) != {0,1} else ('prompt_sha256','token_sha256','prompt_tokens','completion_tokens','cached_tokens','speculative')):
                    assert arms[0][key] == arms[1][key], (n,drafted,rep,key)
                pairs.append(arms)
                write(args.output/'pairs.json',pairs)
    print('PASS: requested cold receipts checked; paired arms match where supplied; physical state bytes still require their own gate')
if __name__ == '__main__': main()
