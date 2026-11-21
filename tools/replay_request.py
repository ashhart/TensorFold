"""List, clear or replay requests from a TENSORFOLD_REQUEST_JOURNAL and compare the reply with the journaled one."""
import argparse
import base64
import copy
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FILES = 4  # the current file and three rotated ones, as request_journal.zig keeps them
QUARANTINE_S = 24 * 60 * 60  # a quarantine lapses a day after its crash


class ReplayError(ValueError):
    pass


def journal_files(path):
    """The kept files oldest first: path.3, path.2, path.1, then path itself."""
    path = Path(path)
    older = [Path(f'{path}.{n}') for n in range(FILES - 1, 0, -1)]
    return [p for p in older if p.exists()] + [path]


def read_journal(path, now=None):
    """Each request's records by id, read as the server reads them; unreadable lines are passed over."""
    entries, pending, crashes, skipped = {}, set(), {}, 0
    now = time.time() if now is None else now
    for file in journal_files(path):
        for line in file.read_bytes().split(b'\n')[:-1]:  # the last piece is empty or a torn line
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            if not isinstance(record, dict) or record.get('version') != 1:
                skipped += 1
                continue
            kind, request_id = record.get('kind'), record.get('id')
            if kind == 'stop':
                pending.clear()
            elif kind == 'clear' and record.get('all'):
                crashes.clear()
            elif kind == 'clear':
                crashes.pop(record.get('fingerprint'), None)
            elif kind == 'start':
                entries[request_id] = {'start': record, 'prepared': None, 'finish': None, 'quarantine': []}
                pending.add(request_id)
            elif kind == 'quarantine' and record.get('at', 0) + QUARANTINE_S > now:
                crashes[record.get('fingerprint')] = crashes.get(record.get('fingerprint'), 0) + 1
            if request_id not in entries or kind not in ('prepared', 'finish', 'quarantine'):
                continue
            entry = entries[request_id]
            if kind == 'prepared':
                entry['prepared'] = record  # the last one made the reply (a compacted request has several)
                continue
            if kind == 'finish':
                entry['finish'] = record
            else:
                entry['quarantine'].append(record)
            pending.discard(request_id)
    for entry in entries.values():
        entry['refused'] = crashes.get(entry['start'].get('fingerprint'), 0) >= 2  # in flight at two crashes
        entry['in_flight'] = entry['start']['id'] in pending
    return entries, skipped


def output_of(raw, stream):
    """What the model produced, without what differs between two runs of one request (ids, times, cache hits)."""
    def merge(into, delta):
        for key, value in delta.items():
            if key in ('id', 'role'):
                continue
            if key == 'tool_calls':
                calls = into.setdefault('tool_calls', {})
                for call in value:
                    merge(calls.setdefault(str(call.get('index', len(calls))), {}), call)
            elif isinstance(value, str) and isinstance(into.get(key), str):
                into[key] += value
            elif isinstance(value, list) and isinstance(into.get(key), list):
                into[key].extend(copy.deepcopy(value))
            elif isinstance(value, dict) and isinstance(into.get(key), dict):
                merge(into[key], value)
            elif value is not None or key not in into:
                into[key] = copy.deepcopy(value)

    def tail(payload, out):
        usage = payload.get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens'):
            if key in usage:
                out[key] = usage[key]
        for key in ('stop_sequence', 'error'):
            if key in payload:
                out[key] = payload[key]
        runtime = payload.get('tensorfold')
        if isinstance(runtime, dict) and 'token_sha256' in runtime:
            out['token_sha256'] = runtime['token_sha256']

    out, choices = {}, {}
    payloads = [json.loads(line[6:]) for line in raw.split(b'\n') if line.startswith(b'data: ') and line != b'data: [DONE]'] if stream else [json.loads(raw)]
    for payload in payloads:
        for choice in payload.get('choices') or []:
            body = {k: v for k, v in choice.items() if k not in ('delta', 'message', 'index')}
            body.update(choice.get('delta') or choice.get('message') or {})
            merge(choices.setdefault(str(choice.get('index', 0)), {}), body)
        tail(payload, out)
    out['choices'] = choices
    return json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def redacted(value):
    if value == '<redacted>':
        return True
    if isinstance(value, dict):
        return any(redacted(v) for v in value.values())
    if isinstance(value, list):
        return any(redacted(v) for v in value)
    return False


def request_body(entry, replacement=None):
    """The journaled body, or its restored copy, with the seed the server drew when the client sent none."""
    body = copy.deepcopy(replacement if replacement is not None else entry['start'].get('body'))
    if body is None:
        raise ReplayError('the body was too large to journal; supply it with --body FILE')
    if not isinstance(body, dict):
        raise ReplayError('the request body must be a JSON object')
    if redacted(body):
        raise ReplayError('the request has redacted media; supply the original with --body FILE')
    prepared = entry['prepared']
    if 'seed' not in body and prepared and prepared.get('seed') is not None:
        body['seed'] = prepared['seed']
    return body


def reference(entry):
    finish = entry['finish']
    if finish is None:
        return None
    raw = finish.get('reply_base64')
    return base64.b64decode(raw, validate=True) if raw is not None else None


def http_send(url, body, key=None):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def replay(entry, base, send, *, replacement=None, whole_body=False, allow_quarantined=False):
    """One send of the request; equal is True, False, or None when the journal holds no reply to compare."""
    request_id = entry['start']['id']
    if entry['quarantine'] and not allow_quarantined:
        raise ReplayError(f'{request_id} was in flight at a crash; pass --allow-quarantined to send it on purpose')
    path = entry['start'].get('path')
    if path not in ('/v1/chat/completions', '/v1/completions'):
        raise ReplayError('the journal names a route replay does not send')
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ReplayError('--base must be an http(s) origin with no credentials, path or query')
    body = request_body(entry, replacement)
    status, actual = send(base.rstrip('/') + path, body)
    finish, expected = entry['finish'], reference(entry)
    result = {'id': request_id, 'actual_status': status, 'actual_bytes': len(actual), 'actual_sha256': hashlib.sha256(actual).hexdigest()}
    if finish is None:
        return dict(result, comparison='no_reference', equal=None), actual
    result.update(expected_status=finish.get('status'), expected_bytes=finish.get('reply_bytes'), expected_sha256=finish.get('reply_sha256'))
    stream = bool(finish.get('stream'))
    if whole_body:
        result['comparison'] = 'whole_body'
        if expected is None:
            result.update(equal=status == finish.get('status') and result['actual_sha256'] == finish.get('reply_sha256'), first_difference=None)
            return result, actual
        left, right = expected, actual
    elif expected is None:
        try:
            tokens = json.loads(output_of(actual, stream)).get('token_sha256')
        except ValueError:
            tokens = None
        result.update(comparison='tokens', equal=status == finish.get('status') and tokens is not None and tokens == finish.get('token_sha256'), first_difference=None)
        return result, actual
    else:
        result['comparison'] = 'output'
        try:
            left, right = output_of(expected, stream), output_of(actual, stream)
        except ValueError:
            left, right = expected, actual
    different = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), None)
    if different is None and len(left) != len(right):
        different = min(len(left), len(right))
    result.update(equal=status == finish.get('status') and left == right, first_difference=different, compared_bytes=[len(left), len(right)])
    return result, actual


def locked(path):
    """True when a server holds the journal's lock; the lock is released when this process exits."""
    fd = os.open(f'{path}.lock', os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return True
    return False


def clear(path, target, entries):
    """Appends a clear record while no server holds the journal; the next start serves those requests again."""
    if locked(path):
        raise ReplayError('a server holds this journal; stop it, clear, then start it again')
    record = {'version': 1, 'kind': 'clear', 'at': int(time.time())}
    if target == 'all':
        record['all'] = True
    else:
        entry = entries.get(target)
        record['fingerprint'] = entry['start']['fingerprint'] if entry else target
        if len(record['fingerprint']) != 64:
            raise ReplayError('name a journaled request id, a 64-character fingerprint, or all')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    try:
        os.write(fd, (json.dumps(record, separators=(',', ':')) + '\n').encode())
    finally:
        os.close(fd)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal')
    parser.add_argument('--list', action='store_true', help='list the journaled requests; nothing is sent')
    parser.add_argument('--clear', metavar='ID|FINGERPRINT|all', help='lift a quarantine (stop the server first)')
    parser.add_argument('--id', help='the request to replay')
    parser.add_argument('--base', help='the server origin, such as http://127.0.0.1:8080')
    parser.add_argument('--body', help='the original request JSON, when its media were redacted or it was too large')
    parser.add_argument('--key-env', default='TENSORFOLD_API_KEY', help='the variable that holds the API key')
    parser.add_argument('--whole-body', action='store_true', help='compare every reply byte, ids and timings included')
    parser.add_argument('--allow-quarantined', action='store_true', help='send a request that was in flight at a crash')
    parser.add_argument('--output', help='write the reply bytes to this file (owner-only)')
    args = parser.parse_args(argv)
    entries, skipped = read_journal(args.journal)
    if skipped:
        print(f'{skipped} unreadable record(s) passed over', file=sys.stderr)
    if args.list:
        rows = [{'id': key, 'at': e['start'].get('at'), 'path': e['start'].get('path'), 'fingerprint': e['start'].get('fingerprint'),
                 'outcome': e['finish'].get('outcome') if e['finish'] else 'in_flight' if e['in_flight'] else 'unfinished',
                 'crashes': len(e['quarantine']), 'refused': e['refused']} for key, e in entries.items()]
        print(json.dumps(rows, indent=2))
        return 0
    if args.clear:
        print(json.dumps(clear(args.journal, args.clear, entries)))
        return 0
    if not args.id or not args.base or args.id not in entries:
        parser.error('name a journaled --id and a --base, or use --list or --clear')
    replacement = json.loads(Path(args.body).read_text()) if args.body else None
    result, actual = replay(entries[args.id], args.base, lambda url, body: http_send(url, body, os.environ.get(args.key_env)), replacement=replacement, whole_body=args.whole_body, allow_quarantined=args.allow_quarantined)
    if args.output:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as out:
            out.write(actual)
    print(json.dumps(result, indent=2))
    return 2 if result['equal'] is None else 0 if result['equal'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ReplayError, OSError, ValueError) as error:
        raise SystemExit(f'replay_request: {error}')
