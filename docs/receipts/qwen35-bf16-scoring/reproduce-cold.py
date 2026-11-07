import argparse, hashlib, json, pathlib, sys
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('fixture');p.add_argument('output');a=p.parse_args()
sys.path.insert(0,a.source+'/tools')
import bench_cache
fixture=json.loads(pathlib.Path(a.fixture).read_text())
ids=fixture['token_ids'] if isinstance(fixture,dict) else fixture
item=fixture['item'] if isinstance(fixture,dict) else {'name':'public-fixture','kind':'completion','prompt':ids}
timeout=3600 if len(ids)>=65536 else 1800
original_urlopen=bench_cache.bench.urllib.request.urlopen
def open_with_timeout(*args,**kwargs):
 kwargs['timeout']=timeout
 return original_urlopen(*args,**kwargs)
bench_cache.bench.urllib.request.urlopen=open_with_timeout
try:
 run=bench_cache.bench.stream('http://127.0.0.1:8091','qwen35-bf16',item,8,0,1234)
finally:bench_cache.bench.urllib.request.urlopen=original_urlopen
row=bench_cache.receipt(run,'first')
report={'schema_version':1,'client':'bench_cache.bench.stream and bench_cache.receipt/qualify; first request only',
 'fixture_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'request_kind':item['kind'],'http_read_timeout_s':timeout,'requested_reply_tokens':8,
 'temperature':0,'seed':1234,'rows':[row],'qualified':bench_cache.qualify([row],len(ids),True,False)}
pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
raise SystemExit(0 if report['qualified'] else 1)
