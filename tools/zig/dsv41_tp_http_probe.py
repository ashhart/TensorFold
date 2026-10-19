"""Compare one-rank and paired native HTTP replies, then verify refusal after peer loss."""
import argparse
import concurrent.futures
import json
import pathlib
import time
import threading
import urllib.error
import urllib.request

PROMPTS = [
    [{"role": "user", "content": "Antworte nur mit dem Ergebnis: 17 + 25 ="}],
    [{"role": "user", "content": "Nenne drei deutsche Farben, durch Kommas getrennt."}],
    [{"role": "user", "content": "Merke dir das Wort Kiefer."},
     {"role": "assistant", "content": "Kiefer."},
     {"role": "user", "content": "Welches Wort war es? Antworte mit einem Wort."}],
]


def payload(index, stream=False):
    return {"model": "DeepSeek-TP-Probe", "messages": PROMPTS[index],
            "temperature": 0, "max_tokens": 32, "stream": stream}


def normalize(body):
    choices = body.get("choices", [])
    assert len(choices) == 1 and choices[0].get("finish_reason"), body
    message = choices[0]["message"]
    assert not message.get("tool_calls"), message
    fields = {key: message.get(key) or "" for key in ("content", "reasoning_content", "reasoning")}
    assert any(fields.values()), "No generated text or reasoning"
    return {"message": fields, "finish_reason": choices[0]["finish_reason"]}


def equal(actual, expected):
    assert actual == expected, {"actual": actual, "expected": expected}


def request(url, data, timeout=180):
    started = time.monotonic()
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    # Network failure is not evidence that a live leader refused a request.
    return {"status": status, "seconds": time.monotonic()-started,
            "body": json.loads(raw)}


def streamed(url, index):
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload(index, True)).encode(),
                                 headers={'Content-Type': 'application/json'})
    fields = {key: "" for key in ("content", "reasoning_content", "reasoning")}
    chunks, reason, done = [], None, False
    first_delta_seconds = None
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        assert response.status == 200
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            assert 'error' not in chunk, chunk
            for choice in chunk.get('choices', []):
                assert choice.get('index', 0) == 0
                delta = choice.get('delta', {})
                assert not delta.get('tool_calls'), delta
                if first_delta_seconds is None and any(delta.get(key) for key in fields):
                    first_delta_seconds = time.monotonic() - started
                for key in fields:
                    fields[key] += delta.get(key) or ''
                if choice.get('finish_reason') is not None:
                    reason = choice['finish_reason']
    assert done and reason and any(fields.values()), (done, reason, fields)
    return {"seconds": time.monotonic()-started, "first_delta_seconds": first_delta_seconds, "chunks": chunks,
            "normalized": {"message": fields, "finish_reason": reason}}


def metrics(url):
    with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=5) as response:
        text = response.read().decode()
    return {line.split()[0]: float(line.split()[1]) for line in text.splitlines()
            if line and not line.startswith('#')}


def cancel_stream(url):
    before = metrics(url)
    data = payload(1, True)
    data['max_tokens'] = 128
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    chunks = []
    with urllib.request.urlopen(req, timeout=180) as response:
        assert response.status == 200
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            value = line[5:].strip()
            assert value != '[DONE]', 'Stream completed before cancellation'
            chunk = json.loads(value)
            chunks.append(chunk)
            assert 'error' not in chunk, chunk
            choices = chunk.get('choices', [])
            assert not any(c.get('finish_reason') for c in choices), 'Reply finished before cancellation'
            if any(any(c.get('delta', {}).get(key) for key in ('content', 'reasoning_content', 'reasoning'))
                   for c in choices):
                break
        else:
            raise AssertionError('No generated delta before EOF')
    closed = time.monotonic()
    samples = []
    while time.monotonic() - closed < 20:
        after = metrics(url)
        samples.append({'seconds': time.monotonic() - closed, 'metrics': after})
        if (after['tensorfold:client_disconnections_total'] > before['tensorfold:client_disconnections_total']
                and after['tensorfold:requests_running'] == 0
                and after['tensorfold:requests_waiting'] == 0):
            return {'before': before, 'partial_chunks': chunks, 'after_close': samples,
                    'server_observed_disconnect': True}
        time.sleep(0.1)
    return {'before': before, 'partial_chunks': chunks, 'after_close': samples,
            'server_observed_disconnect': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'parity', 'refuse'])
    parser.add_argument('url')
    parser.add_argument('out', type=pathlib.Path)
    parser.add_argument('--baseline', type=pathlib.Path)
    parser.add_argument('--follower')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    receipts = []

    def save(name, value):
        (args.out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
        receipts.append(name)
        (args.out / 'progress.json').write_text(json.dumps({'completed': receipts}) + '\n')

    if args.mode == 'refuse':
        for i in range(2):
            result = request(args.url, payload(0), timeout=35)
            save('refusal-' + str(i), result)
            assert result['status'] >= 400 and not result['body'].get('choices'), result
            assert 'restart both ranks' in json.dumps(result['body']).lower(), result
            assert result['seconds'] < (35 if i == 0 else 5), result
    else:
        expected = []
        if args.mode == 'parity':
            assert args.baseline and args.follower
            prior = json.loads((args.baseline / 'result.json').read_text())
            assert prior['state'] == 'passed' and prior['mode'] == 'baseline', prior
            expected = json.loads((args.baseline / 'expected.json').read_text())
            assert len(expected) == len(PROMPTS)
            follower = request(args.follower, payload(0), timeout=35)
            save('follower-refusal', follower)
            assert follower['status'] >= 400 and not follower['body'].get('choices'), follower
            assert 'follows rank 0' in json.dumps(follower['body']).lower(), follower
        solo, benchmark = [], []
        for repeat in range(3):
            for i in range(len(PROMPTS)):
                result = request(args.url, payload(i))
                save(('solo-' if repeat == 0 else 'repeat-' + str(repeat) + '-solo-') + str(i), result)
                assert result['status'] == 200, result
                current = normalize(result['body'])
                if expected:
                    equal(current, expected[i])
                if repeat == 0:
                    solo.append(current)
                else:
                    equal(current, solo[i])
                benchmark.append({'repeat': repeat, 'prompt': i, 'seconds': result['seconds'], 'usage': result['body'].get('usage')})
            sample = streamed(args.url, 0)
            save('repeat-' + str(repeat) + '-stream', sample)
            equal(sample['normalized'], solo[0])
        save('benchmark', benchmark)
        save('expected', solo)
        if args.mode == 'parity':
            # Four repetitions per prompt; two workers match the server's two lane slots.
            order = [0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 2, 1]
            samples, errors = [], []
            stop = threading.Event()

            def observe():
                while not stop.is_set():
                    try:
                        with urllib.request.urlopen(args.url.rstrip('/') + '/metrics', timeout=3) as response:
                            text = response.read().decode()
                        values = {line.split()[0]: float(line.split()[1]) for line in text.splitlines()
                                  if line and not line.startswith('#')}
                        samples.append({'at': time.monotonic(),
                                        'running': values['tensorfold:requests_running'],
                                        'waiting': values['tensorfold:requests_waiting']})
                    except Exception as exc:
                        errors.append(type(exc).__name__ + ': ' + str(exc))
                    stop.wait(0.1)

            monitor = threading.Thread(target=observe, daemon=True)
            monitor.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(request, args.url, payload(i)) for i in order]
                    for n, (i, future) in enumerate(zip(order, futures)):
                        result = future.result()
                        save('concurrent-' + str(n), result)
                        assert result['status'] == 200, result
                        equal(normalize(result['body']), solo[i])
            finally:
                stop.set()
                monitor.join(timeout=5)
                save('concurrency-observation', {'samples': samples, 'errors': errors})
            assert not monitor.is_alive() and not errors, errors
            assert samples and max(x['running'] for x in samples) >= 2, 'No evidence of two admitted requests'
            stream = streamed(args.url, 0)
            save('stream', stream)
            equal(stream['normalized'], solo[0])
            cancelled = cancel_stream(args.url)
            save('cancelled-stream', cancelled)
            assert cancelled['server_observed_disconnect'], 'No server-side cancellation evidence'
            last = request(args.url, payload(0))
            save('after-cancelled-stream', last)
            assert last['status'] == 200
            equal(normalize(last['body']), solo[0])
    save('result', {'state': 'passed', 'mode': args.mode, 'scope': receipts[:]})
    print('HTTP_PROBE_PASSED', args.mode, flush=True)


if __name__ == '__main__':
    main()
