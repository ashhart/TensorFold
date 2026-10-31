import json,http.client,time,concurrent.futures,base64,io
from pathlib import Path
from PIL import Image
from features import checks

def run(out,name):
 result={'basic':checks(out,name)}
 thinking=json.load(open(out/f'{name}-feature-thinking.json'))['response']
 assert thinking['choices'][0]['message'].get('reasoning_content'),'No observed reasoning output'
 result['reasoning_observed']=True
 def call(label,body,timeout=600):
  body={**body,'model':'qwen-community','stream':False,'return_token_ids':True,'priority':'background'}
  c=http.client.HTTPConnection('127.0.0.1',18080,timeout=timeout);start=time.monotonic();c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'});r=c.getresponse();raw=r.read();c.close()
  if r.status!=200:raise RuntimeError((label,r.status,raw[:300]))
  x=json.loads(raw);(out/f'{name}-{label}.json').write_text(json.dumps({'request':body,'response':x,'elapsed_s':time.monotonic()-start},indent=2));return x
 base={'messages':[{'role':'user','content':'Explain how a hash table handles collisions.'}],'temperature':0,'seed':42,'max_tokens':128,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
 structured=call('structured',{**base,'messages':[{'role':'user','content':'Return the integer answer 42 in the requested JSON object.'}],'response_format':{'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':{'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer'],'additionalProperties':False}}}})
 assert json.loads(structured['choices'][0]['message']['content'])=={'answer':42};result['structured']=structured['tensorfold']['token_sha256']
 original=json.load(open('/task/png-request.json'))
 url=next(p['image_url']['url'] for p in original['messages'][0]['content'] if p['type']=='image_url');im=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).convert('RGB')
 for fmt,mime in [('JPEG','jpeg'),('WEBP','webp')]:
  buf=io.BytesIO();im.save(buf,format=fmt);body=json.loads(json.dumps(original))
  for p in body['messages'][0]['content']:
   if p['type']=='image_url':p['image_url']['url']='data:image/'+mime+';base64,'+base64.b64encode(buf.getvalue()).decode()
  x=call('vision-'+mime,body);assert 'red' in (x['choices'][0]['message']['content'] or '').lower();result['vision-'+mime]=x['tensorfold']['token_sha256']
 # Queueing at parallel=1 must preserve each request's solo output.
 bodies=[{**base,'messages':[{'role':'user','content':p}],'max_tokens':64} for p in ['Explain a binary search tree.','Write a Python function that adds two integers.']]
 solo=[call('solo-'+str(i),b)['tensorfold']['token_sha256'] for i,b in enumerate(bodies)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  replies=list(pool.map(lambda ib:call('concurrent-'+str(ib[0]),ib[1]),enumerate(bodies)))
 assert [x['tensorfold']['token_sha256'] for x in replies]==solo;result['concurrent']=solo
 # Close a streaming response early, then confirm the next full request still matches.
 c=http.client.HTTPConnection('127.0.0.1',18080,timeout=120);c.request('POST','/v1/chat/completions',json.dumps({**base,'model':'qwen-community','stream':True,'max_tokens':2048,'priority':'background'}),{'Content-Type':'application/json'});r=c.getresponse();assert r.status==200;lines=[]
 while len(lines)<2:
  line=r.readline()
  if not line:break
  if line.startswith(b'data:'):lines.append(line.decode(errors='replace'))
 r.close();c.close();time.sleep(.5);after=call('after-cancel',base);assert after['tensorfold']['token_sha256']==result['basic']['draft'];result['cancellation_recovery']=True
 (out/f'{name}-cancel.json').write_text(json.dumps({'early_events':lines,'recovery_sha':after['tensorfold']['token_sha256']},indent=2))
 # Prefix reuse and longer contexts, including near the 128K configured limit.
 result['contexts']={}
 for size in ['8192','32768','near128k']:
  body=json.load(open('/task/long-'+size+'.json'));body['max_tokens']=64
  cold=call('context-'+size+'-cold',body);warm=call('context-'+size+'-warm',body)
  assert cold['usage']['prompt_tokens_details']['cached_tokens']==0
  assert warm['usage']['prompt_tokens_details']['cached_tokens']>0
  assert cold['tensorfold']['token_sha256']==warm['tensorfold']['token_sha256']
  if size=='near128k':assert 120000<cold['usage']['prompt_tokens']<131008
  result['contexts'][size]={'prompt_tokens':cold['usage']['prompt_tokens'],'cached_tokens':warm['usage']['prompt_tokens_details']['cached_tokens'],'hash':cold['tensorfold']['token_sha256']}
 return result
