import http.client,json,hashlib
from pathlib import Path

def checks(out, name):
    def call(label, body):
        body={**body,'model':'qwen-community','return_token_ids':True,'stream':False,'priority':'background'}
        c=http.client.HTTPConnection('127.0.0.1',18080,timeout=300)
        c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
        r=c.getresponse();raw=r.read();c.close()
        assert r.status==200,(label,r.status,raw[:200])
        x=json.loads(raw)
        (out/f'{name}-feature-{label}.json').write_text(json.dumps({'request':body,'response':x},indent=2))
        return x
    base={'messages':[{'role':'user','content':'Explain how a hash table handles collisions.'}],'temperature':0,'seed':42,'max_tokens':128,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
    plain=call('plain',{**base,'draft':False});draft=call('draft',{**base,'draft':True})
    assert plain['tensorfold']['token_sha256']==draft['tensorfold']['token_sha256'],'Draft/plain mismatch'
    png=call('png',json.loads(Path('/task/png-request.json').read_text()))
    assert 'red' in (png['choices'][0]['message']['content'] or '').lower(),'Vision answer failed'
    tool=call('tool',{**base,'messages':[{'role':'user','content':'Get the weather for Berlin.'}], 'tools':[{'type':'function','function':{'name':'weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}], 'tool_choice':'required'})
    assert tool['choices'][0]['message'].get('tool_calls'),'Required tool failed'
    think=call('thinking',{**base,'reasoning_effort':'low','chat_template_kwargs':{'enable_thinking':True},'messages':[{'role':'user','content':'What is 19 times 23?'}]})
    assert think['tensorfold']['enable_thinking'],'Thinking not enabled'
    return {label:r['tensorfold']['token_sha256'] for label,r in [('plain',plain),('draft',draft),('png',png),('tool',tool),('thinking',think)]}
