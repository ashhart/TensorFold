"""Short fixed-output-cap concurrent decode receipts, independent of user traffic."""
import argparse
import concurrent.futures
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--base', default='http://127.0.0.1:8890/v1')
a = p.parse_args()


def run(i):
    body = dict(model='glm-5.3-flash-fp4-local', temperature=1, top_p=.95,
                seed=119+i, reasoning_effort='low', max_tokens=256,
                messages=[dict(role='user', content=f'测试编号{i}。请用中文写一份家用服务器的数据备份操作指南，说明独立副本、恢复演练和校验三个方面。')])
    req = urllib.request.Request(a.base+'/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        result = json.load(r)
    assert len(result['tensorfold']['token_sha256']) == 64
    return dict(request=body, result=result)


for n in (2, 4):
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        rows = list(pool.map(run, range(n)))
    elapsed = time.monotonic()-start
    print(json.dumps(dict(label=a.label, concurrency=n, wall=elapsed,
                          total_tokens=sum(r['result']['usage']['completion_tokens'] for r in rows),
                          requests=rows), ensure_ascii=False), flush=True)
