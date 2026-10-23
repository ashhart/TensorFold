#!/usr/bin/env python3
"""Non-destructive qualification against the isolated test port; emits JSONL receipts."""
import argparse, base64, concurrent.futures, hashlib, json, struct, time, urllib.request, zlib

p = argparse.ArgumentParser()
p.add_argument('--base', default='http://127.0.0.1:8890/v1')
p.add_argument('--model', default='glm-5.3-flash-fp4-local')
p.add_argument('--mode', choices=['smoke','parallel','vision','long','cancel','cancel-long','sampled','capacity','parallel-long','api'], default='smoke')
a = p.parse_args()

def post(path, body):
    started = time.monotonic()
    req = urllib.request.Request(a.base+path, json.dumps(body).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        result = json.load(r)
    result['_wall'] = round(time.monotonic()-started,3)
    return result

def chat(prompt, **kw):
    body = dict(model=a.model, messages=[dict(role='user',content=prompt)], temperature=0,
                seed=119, reasoning_effort='low', max_tokens=384)
    body.update(kw)
    return post('/chat/completions', body)

def receipt(name, result):
    print(json.dumps(dict(test=name, result=result),ensure_ascii=False),flush=True)
    if 'equivalence' in name:
        assert all(v for k,v in result.items() if k.endswith('equal')), (name,result)
    return result

def sig(r):
    runtime = r.get('tensorfold',{})
    value = runtime.get('token_sha256') or runtime.get('token_sha')
    if not isinstance(value, str) or len(value) not in (12,64):
        raise AssertionError('Missing valid token hash; equality cannot be established')
    return value

def image():
    def chunk(k,d):
        return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d)&0xffffffff)
    # A plain red square, with no external source or image editing dependency.
    raw = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))
    raw += chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*64)*64))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(raw).decode()

if a.mode == 'smoke':
    for i,prompt in enumerate(['请计算17乘以19，只用一句中文回答。','用中文简要说明为什么太阳会发光。']):
        yes=receipt('drafted-'+str(i),chat(prompt))
        no=receipt('serial-'+str(i),chat(prompt,draft=False))
        receipt('draft-equivalence-'+str(i),dict(equal=sig(yes)==sig(no),draft_sha=sig(yes),serial_sha=sig(no)))
    tool=dict(type='function',function=dict(name='get_weather',description='查询城市天气',
              parameters=dict(type='object',properties={'city':{'type':'string'}},required=['city'],additionalProperties=False)))
    receipt('required-tool',chat('请查询北京的天气。',tools=[tool],tool_choice='required'))
    receipt('responses',post('/responses',dict(model=a.model,input='用一句中文介绍自己。',
              reasoning={'effort':'low'},include=['reasoning.encrypted_content'],max_output_tokens=384,temperature=0,seed=119)))
elif a.mode == 'parallel':
    prompts=[f'编号{i}：用两句话解释云为什么会下雨。' for i in range(4)]
    solo=[receipt('solo-'+str(i),chat(x)) for i,x in enumerate(prompts)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        concurrent=list(pool.map(chat,prompts))
    for i,r in enumerate(concurrent):
        receipt('parallel-'+str(i),r)
        receipt('parallel-equivalence-'+str(i),dict(equal=sig(r)==sig(solo[i]),solo_sha=sig(solo[i]),parallel_sha=sig(r)))
elif a.mode == 'vision':
    url=image()
    receipt('image',chat([{'type':'text','text':'这张图片的主要颜色是什么？用中文回答。'},
                          {'type':'image_url','image_url':{'url':url}}]))
    receipt('tool-output-image',post('/responses',dict(model=a.model,temperature=0,seed=119,
        reasoning={'effort':'low'},max_output_tokens=384,input=[
        {'type':'message','role':'user','content':'请查看工具提供的图片并说明主要颜色。'},
        {'type':'function_call','name':'get_picture','arguments':'{}','call_id':'color_test'},
        {'type':'function_call_output','call_id':'color_test','output':[
            {'type':'input_text','text':'这是工具返回的图片。'}, {'type':'input_image','image_url':url}]}])))
elif a.mode == 'long':
    text=''.join(f'记录{i:05d}：这是一条用于缓存和长上下文一致性检验的普通记录。\n' for i in range(2200))
    prompt='请阅读下面资料，最后只回答指定校验码。\n'+text+'\n最终校验码是：GX10-7391。请准确复述校验码。'
    one=receipt('long-cold',chat(prompt,max_tokens=192))
    assert one.get('usage',{}).get('prompt_tokens',0) > 32768, 'Long-context test was too short'
    two=receipt('long-reused',chat(prompt,max_tokens=192))
    no=receipt('long-serial',chat(prompt,max_tokens=192,draft=False))
    receipt('long-equivalence',dict(repeated_equal=sig(one)==sig(two),serial_equal=sig(one)==sig(no),
            cached_tokens=two.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0)))
elif a.mode == 'cancel':
    prompt='请连续写出从1到1000的整数，用逗号分隔。'
    body=dict(model=a.model,messages=[dict(role='user',content=prompt)],stream=True,max_tokens=2048,
              temperature=0,seed=119,reasoning_effort='low')
    req=urllib.request.Request(a.base+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:
        for _ in range(20):
            if not r.readline(): break
    receipt('cancel-sent',{'closed_after_lines':20})
    for i in range(2): receipt('after-cancel-'+str(i),chat('请计算17乘以19，只用一句中文回答。'))
elif a.mode == 'cancel-long':
    text=''.join(f'记录{i:05d}：这是一条用于缓存和长上下文一致性检验的普通记录。\n' for i in range(2200))
    prompt=('请阅读下面的公开合成记录。\n'+text+
            '\n请用中文详细说明长上下文服务中取消、重试与缓存复用的关系。'+
            '至少分十二节展开，每节解释一个具体场景、潜在问题及检查方法，全文不少于三千字。')
    settings=dict(temperature=1,top_p=.95,seed=119,reasoning_effort='low')
    body=dict(model=a.model,messages=[dict(role='user',content=prompt)],stream=True,
              max_tokens=2048,draft=True,**settings)
    req=urllib.request.Request(a.base+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    lines=0
    generated_chunks=0
    finished=False
    with urllib.request.urlopen(req,timeout=7200) as response:
        for _ in range(20):
            raw=response.readline()
            if not raw: break
            lines+=1
            line=raw.decode().strip()
            if not line.startswith('data:'): continue
            payload=line[5:].strip()
            if payload == '[DONE]':
                finished=True
                break
            event=json.loads(payload)
            if 'error' in event: raise AssertionError(event)
            for choice in event.get('choices',[]):
                finished=finished or choice.get('finish_reason') is not None
                delta=choice.get('delta',{})
                if any(delta.get(k) for k in ('content','reasoning_content','reasoning')):
                    generated_chunks+=1
            if finished: break
    receipt('cancel-long-sent',dict(closed_after_lines=lines,generated_chunks=generated_chunks,
            completed_before_close=finished,max_tokens=2048,**settings))
    assert lines == 20 and generated_chunks > 0 and not finished, 'Stream was not cancelled during generation'
    yes=receipt('cancel-long-retry-drafted',chat(prompt,max_tokens=128,draft=True,**settings))
    no=receipt('cancel-long-retry-serial',chat(prompt,max_tokens=128,draft=False,**settings))
    for result in (yes,no):
        assert result.get('usage',{}).get('prompt_tokens',0) > 32768, 'Cancellation retry context was too short'
    receipt('cancel-long-equivalence',dict(equal=sig(yes)==sig(no),draft_sha=sig(yes),serial_sha=sig(no),
            prompt_tokens=yes['usage']['prompt_tokens'],seed=settings['seed'],temperature=1,top_p=.95))
elif a.mode == 'sampled':
    for seed in (119, 902):
        prompt='写一个两句话的微型科幻故事，包含一只猫和一艘飞船。'
        yes=receipt('sampled-draft-'+str(seed),chat(prompt,temperature=1,top_p=.95,seed=seed,max_tokens=256))
        no=receipt('sampled-serial-'+str(seed),chat(prompt,temperature=1,top_p=.95,seed=seed,max_tokens=256,draft=False))
        receipt('sampled-equivalence-'+str(seed),dict(equal=sig(yes)==sig(no),draft_sha=sig(yes),serial_sha=sig(no)))
elif a.mode == 'capacity':
    def make_prompt(n):
        return ('请阅读以下记录，最后只回答指定校验码。\n'+
                ''.join(f'记录{i:05d}：这是一条用于大容量上下文检查的普通记录。\n' for i in range(n))+
                '\n最终校验码是：GX10-512K-8426。请准确复述校验码。')
    n=27000
    for _ in range(5):
        prompt=make_prompt(n)
        tokenized=post('/tokenize',dict(model=a.model,messages=[dict(role='user',content=prompt)],reasoning_effort='low'))
        count=tokenized.get('count') or tokenized.get('prompt_tokens') or len(tokenized.get('tokens',[]))
        if not isinstance(count,int) or count <= 0: raise AssertionError(str(tokenized)[:1000])
        receipt('capacity-token-count',dict(records=n,tokens=count))
        if 518000 <= count <= 522000: break
        n=max(1,int(n*520000/count))
    assert 518000 <= count <= 522000, count
    result=receipt('capacity-512k',chat(prompt,max_tokens=128))
    assert result['usage']['prompt_tokens'] >= 518000
    assert 'GX10-512K-8426' in result['choices'][0]['message']['content']
    receipt('capacity-post-check',chat('请计算17乘以19，只用一句中文回答。'))
elif a.mode == 'parallel-long':
    common=''.join(f'记录{i:05d}：这是一条用于四并发与长上下文检查的普通记录。\n' for i in range(2200))
    def run(i):
        return chat(f'独立请求编号{i}。\n'+common+f'\n请只回答最后校验码：GX10-C{i}-7391。',max_tokens=64)
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results=list(pool.map(run,range(4)))
    for i,r in enumerate(results):
        receipt('parallel-long-'+str(i),r)
        assert r['usage']['prompt_tokens'] > 32768
        assert f'GX10-C{i}-7391' in r['choices'][0]['message']['content']
    receipt('parallel-long-wall',dict(seconds=round(time.monotonic()-started,3)))
elif a.mode == 'api':
    result=receipt('json-schema',chat('计算17加29，按照要求的JSON格式回答。',response_format={
        'type':'json_schema','json_schema':{'name':'sum','strict':True,'schema':{
            'type':'object','properties':{'result':{'type':'integer'}},'required':['result'],'additionalProperties':False}}}))
    assert json.loads(result['choices'][0]['message']['content']) == {'result':46}
    body=dict(model=a.model,input='请计算17乘以19，只用一句中文回答。',stream=True,
              max_output_tokens=128,temperature=0,seed=119,reasoning={'effort':'low'},include=['reasoning.encrypted_content'])
    req=urllib.request.Request(a.base+'/responses',json.dumps(body).encode(),{'Content-Type':'application/json'})
    events=[]
    with urllib.request.urlopen(req,timeout=180) as response:
        for raw in response:
            line=raw.decode().strip()
            if line.startswith('data: ') and line[6:] != '[DONE]': events.append(json.loads(line[6:]))
    receipt('responses-stream',dict(events=events))
    assert any(x.get('type') == 'response.completed' for x in events)
    assert '323' in ''.join(x.get('delta','') for x in events if x.get('type') == 'response.output_text.delta')
