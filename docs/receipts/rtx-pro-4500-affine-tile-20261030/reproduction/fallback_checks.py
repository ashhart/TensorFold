import http.client,json

def run(out,name):
 base={'model':'qwen-community','messages':[{'role':'user','content':'Write a detailed paragraph about hash tables and collision handling.'}],'temperature':0,'seed':42,'max_tokens':96,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False},'return_token_ids':True,'priority':'background'}
 result={}
 for label,extra in [('penalties',{'presence_penalty':0.5,'frequency_penalty':0.1}),('logprobs',{'logprobs':True,'top_logprobs':5}),('sample-filters',{'temperature':0.7,'top_k':40,'top_p':0.85,'min_p':0.05})]:
  body={**base,**extra};c=http.client.HTTPConnection('127.0.0.1',18080,timeout=300);c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'});r=c.getresponse();raw=r.read();c.close()
  if label=='logprobs' and r.status==400:
   assert 'logprobs are not supported' in raw.decode();result[label]={'status':400,'unchanged_refusal':True};(out/f'{name}-fallback-{label}.json').write_text(json.dumps({'request':body,'status':400,'response':json.loads(raw)},indent=2));continue
  assert r.status==200,(label,r.status,raw[:300]);x=json.loads(raw)
  (out/f'{name}-fallback-{label}.json').write_text(json.dumps({'request':body,'response':x},indent=2));result[label]=x['tensorfold']['token_sha256']
  if label=='logprobs':assert x['choices'][0].get('logprobs')
 return result
