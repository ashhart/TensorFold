"""Controlled local public fixtures; JSONL includes exact requests and token hashes."""
import argparse
import hashlib
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--base', default='http://127.0.0.1:8890/v1')
p.add_argument('--label', required=True)
p.add_argument('--reps', type=int, default=2)
p.add_argument('--mode', choices=['decode', 'prefill', 'all'], default='all')
p.add_argument('--warm-prefill', action='store_true', help='Record a separate untimed-for-comparison prompt-kernel warmup')
a = p.parse_args()
model = 'glm-5.3-flash-fp4-local'


def post(path, body):
    req = urllib.request.Request(a.base + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        result = json.load(r)
    return result, time.monotonic() - start


def emit(cell, rep, body):
    result, wall = post('/chat/completions', body)
    runtime = result.get('tensorfold', {})
    assert len(runtime.get('token_sha256', '')) == 64, result
    assert runtime.get('enable_thinking') is True, result
    print(json.dumps(dict(label=a.label, cell=cell, rep=rep, request=body,
                          wall=wall, result=result), ensure_ascii=False), flush=True)
    return result


def body(prompt, temperature=1, seed=119, tokens=384):
    return dict(model=model, messages=[dict(role='user', content=prompt)],
                temperature=temperature, top_p=.95, seed=seed,
                reasoning_effort='low', max_tokens=tokens)


if a.warm_prefill:
    emit('prefill-warmup', 0, body('热身资料，请只回答已阅读。\n' +
         ''.join(f'记录{i:05d}：这是一条用于缓存和长上下文一致性检验的普通记录。\n' for i in range(110)), tokens=8))

if a.mode in ('decode', 'all'):
    prose = '请用中文解释操作系统的虚拟内存、物理内存和swap之间的关系，并举两个例子说明为什么swap使用量不等于持续换页。'
    code = '生成一个完整的单文件HTML页面，介绍一台家用AI服务器，包括深色主题、配置表、三个功能卡片和响应式布局。只输出HTML代码。'
    emit('warmup', 0, body('请用一句中文解释缓存。', tokens=64))
    for rep in range(a.reps):
        for temp in (0, 1):
            for name, prompt in [('prose', prose), ('code', code)]:
                emit(name + '-t' + str(temp), rep, body(prompt, temp))

if a.mode in ('prefill', 'all'):
    for n in (110, 440, 1760, 3520):
        prompt = ('本次独立资料长度为' + str(n) + '条。请阅读资料后解释三条备份原则。\n' +
                  ''.join(f'记录{i:05d}：这是一条用于缓存和长上下文一致性检验的普通记录。\n' for i in range(n)) +
                  '\n请说明为什么备份需要独立副本、恢复演练和校验，约100字。')
        request = body(prompt, 1, tokens=128)
        ids, _ = post('/tokenize', {k: v for k, v in request.items() if k in ('model', 'messages', 'reasoning_effort')})
        print(json.dumps(dict(label=a.label, cell='prompt-ids-' + str(n), result=ids)), flush=True)
        result = emit('cold-' + str(n), 0, request)
        assert result['usage']['prompt_tokens_details']['cached_tokens'] == 0, 'Not a cold prompt'
