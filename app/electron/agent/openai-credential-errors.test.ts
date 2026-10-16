import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { streamChat, quickComplete } from './openai'
import { runAgent } from './loop'
import { setEngineServeHook } from './engine/provider'
import { createEmitHelpers } from '../main/host/emit-helpers'
import { createObserverHub } from './observer-hooks'
import { classifyModelExhaustion } from './model-router'
import { store } from '../main/state'

// Explicitly fictional credential, including JSON/URL escaping characters.
const key = 'FICTIONAL_ONLY_credential+"quote"/back\\slash%=5837'
const variants = [key, JSON.stringify(key).slice(1,-1), JSON.stringify(key).slice(1,-1).replace(/\//g,'\\/'), encodeURIComponent(key), encodeURIComponent(key).toLowerCase().replace('fictional_only','FICTIONAL_ONLY')]
let mode = 'http', encoding = key
const sent: string[] = [], requests: any[] = []
const server = createServer((request, response) => {
  let raw = ''; request.on('data', (chunk) => { raw += chunk; assert.ok(raw.length < 200000) }); request.on('end', () => {
    requests.push(JSON.parse(raw)); const credential = String(request.headers.authorization ?? '').slice(7); sent.push(credential)
    const detail = `HTTP 401 synthetic diagnostic. Available models: alpha, beta. Echo ${encoding}`
    if (mode === 'http' || mode === 'http-limit' || mode === 'http-unicode') { response.writeHead(mode === 'http-limit' ? 429 : 401, `Synthetic ${credential}`, { 'content-type':'application/json','retry-after':'17','x-ratelimit-reset':'2000000000','x-fixture-echo':credential }); const body=JSON.stringify({error:{message:detail}});response.end(mode==='http-unicode'?body.replace('FICTIONAL','\\u0046ICTIONAL'):body); return }
    if (mode === 'json-malformed') { response.writeHead(200,{'content-type':'application/json'});response.end(credential);return }
    if (mode === 'json-abort') { response.writeHead(200,{'content-type':'application/json'});response.flushHeaders();return }
    if (mode === 'json-normal') { response.writeHead(200,{'content-type':'application/json'}); response.end(JSON.stringify({choices:[{message:{content:'Ordinary non-secret assistant text.'},finish_reason:'stop'}]})); return }
    if (mode === 'json-cross-channel') { const at=Math.floor(encoding.length/2);response.writeHead(200,{'content-type':'application/json'});response.end(JSON.stringify({choices:[{message:{reasoning_content:encoding.slice(0,at),content:encoding.slice(at)},finish_reason:'stop'}]}));return }
    if (mode === 'json-error') { response.writeHead(200,{'content-type':'application/json'}); response.end(JSON.stringify({error:{message:detail}})); return }
    if (mode === 'json-finish-error') { response.writeHead(200,{'content-type':'application/json'}); response.end(JSON.stringify({choices:[{message:{content:credential,reasoning_content:credential},error:{message:detail},finish_reason:'error'}]})); return }
    response.writeHead(200,{'content-type':'text/event-stream'})
    const frame = (value: unknown) => response.write(`data: ${JSON.stringify(value)}\n\n`)
    if (mode === 'metadata' || mode === 'metadata-clean' || mode === 'metadata-form' || mode === 'metadata-host') { const ordinary='https://example.invalid/ordinary?q=green';frame({choices:[{delta:{content:'Ordinary metadata answer.'}}]});frame({choices:[{delta:{},finish_reason:'stop'}],usage:{completion_tokens:4,prompt_tokens:9,prompt_tokens_details:{cached_tokens:5}},tensorfold:{tokens_per_second:15,seconds:2,prefill_seconds:1,time_to_first_token:0.1},cortheon:{task:mode!=='metadata-clean'?credential:'Ordinary research',searches:2,pages_read:3,sources:[ordinary,mode==='metadata-host'?`https://%${credential.charCodeAt(0).toString(16)}${credential.slice(1)}.example.invalid/path`:mode==='metadata-form'?`https://example.invalid/research?${new URLSearchParams({token:credential})}`:mode==='metadata'?`https://example.invalid/research?token=${encodeURIComponent(credential)}`:'https://example.invalid/second?q=blue']}});response.end('data: [DONE]\n\n');return }
    if(mode==='finish-key'){frame({choices:[{delta:{content:'Ordinary finish answer.'},finish_reason:credential}]});response.end('data: [DONE]\n\n');return}
    if(mode==='tool-id-key'||mode==='tool-name-key'||mode==='tool-args-positive'){frame({choices:[{delta:{tool_calls:[{index:0,id:mode==='tool-id-key'?credential:'ordinary-call',type:'function',function:{name:mode==='tool-name-key'?credential:'synthetic_probe',arguments:JSON.stringify({value:mode==='tool-args-positive'?credential:'ordinary'})}}]},finish_reason:'tool_calls'}]});response.end('data: [DONE]\n\n');return}
    if (mode === 'normal') { for (const content of ['Ordinary ','non-secret ','assistant text.']) frame({choices:[{delta:{content}}]}); frame({choices:[{delta:{},finish_reason:'stop'}]}); response.end('data: [DONE]\n\n'); return }
    if (mode === 'cross-channel-stop' || mode === 'cross-channel-error') {const at=Math.floor(encoding.length/2);frame({choices:[{delta:{content:encoding.slice(0,at)}}]});frame({choices:[{delta:{reasoning_content:encoding.slice(at)}}]});if(mode==='cross-channel-stop'){frame({choices:[{delta:{},finish_reason:'stop'}]});response.end('data: [DONE]\n\n');return}}
    if (mode === 'split') { const at=Math.floor(encoding.length/2); frame({choices:[{delta:{content:`Safe text ${encoding.slice(0,at)}`}}]}); frame({choices:[{delta:{content:`${encoding.slice(at)} end.`}}]}) }
    if (mode === 'split-reasoning') { const at=Math.floor(encoding.length/2); frame({choices:[{delta:{reasoning_content:encoding.slice(0,at)}}]}); frame({choices:[{delta:{reasoning_content:encoding.slice(at)}}]}) }
    frame(mode === 'sse-error' ? {error:{message:detail}} : {choices:[{delta:{content:credential,reasoning_content:credential},error:{message:detail},finish_reason:'error'}]})
    response.end('data: [DONE]\n\n')
  })
})
await new Promise<void>((resolve)=>server.listen(0,'127.0.0.1',resolve))
const provider:any={id:'synthetic',name:'Synthetic loopback API',kind:'local',transport:'chat-completions',baseUrl:`http://127.0.0.1:${(server.address() as any).port}/v1`,apiKey:key,models:['synthetic']}
const params:any={provider,baseUrl:provider.baseUrl,apiKey:key,model:'synthetic',messages:[{role:'user',content:'Synthetic fixed prompt.'}],approvalMode:'readonly',signal:new AbortController().signal}
function inspect(value: unknown): string { const error=value as any; return JSON.stringify({message:error?.message,stack:error?.stack,own:error?Object.fromEntries(Object.entries(error).map(([name,v])=>[name,v instanceof Headers?Object.fromEntries(v.entries()):v])):value}) }
function noCredential(text: string) { for(const candidate of variants) assert.ok(!text.includes(candidate), 'request credential or its common escaped form must not reach error/output/projection') }
const saved={win:store.win,threads:store.threads,jobs:store.jobs,webhooks:store.webhooks}
try {
  for (const selected of ['http','http-limit','http-unicode','sse-error','sse-finish-error','json-error','json-finish-error','split','split-reasoning','cross-channel-error']) for (const encoded of variants) {
    mode=selected;encoding=encoded;const chunks:any[]=[];let failure:any
    try {for await(const chunk of streamChat(params)) chunks.push(chunk)} catch(error){failure=error}
    assert.ok(failure instanceof Error, `provider ${mode} failure must not appear as blank success`)
    noCredential(inspect(failure));noCredential(JSON.stringify(chunks));noCredential(chunks.filter(chunk=>chunk.type==='content'||chunk.type==='reasoning').map(chunk=>chunk.text).join(''))
    const jsonAt=failure.message.indexOf(': {')
    if(jsonAt>=0)noCredential(JSON.parse(failure.message.slice(jsonAt+2).trim()).error.message)
    assert.match(failure.message,/alpha, beta/,'useful model diagnostics survive secret scrubbing')
    if(mode==='http'||mode==='http-limit'||mode==='http-unicode'){const http = failure as Error & { status:number;headers:Headers };assert.equal(http.status,mode==='http-limit'?429:401);assert.equal(http.headers.get('retry-after'),'17');assert.equal(http.headers.get('x-ratelimit-reset'),'2000000000');assert.ok(!http.headers.get('x-fixture-echo')!.includes(key));if(mode==='http-limit'){const limit=classifyModelExhaustion(http,1000);assert.equal(limit?.retryAfterMs,17000);assert.equal(limit?.resetAt,18000)}}
  }
  for (const selected of ['cross-channel-stop','json-cross-channel']) {mode=selected;encoding=key;const chunks:any[]=[];for await(const chunk of streamChat(params))chunks.push(chunk);noCredential(chunks.filter(chunk=>chunk.type==='content'||chunk.type==='reasoning').map(chunk=>chunk.text).join(''))}
  mode='json-normal';assert.equal(await quickComplete(params),'Ordinary non-secret assistant text.')
  mode='normal';const chunks:any[]=[];for await(const chunk of streamChat(params)) chunks.push(chunk)
  assert.equal(chunks.filter(chunk=>chunk.type==='content').map(chunk=>chunk.text).join(''),'Ordinary non-secret assistant text.','normal non-credential assistant text is unchanged')
  for(const selected of ['json-error','json-finish-error']) { mode=selected;encoding=key;await assert.rejects(()=>quickComplete(params),(error:any)=>{noCredential(inspect(error));return /alpha, beta/.test(error.message)}) }

  mode='http';encoding=key
  const padded=`  ${key}  `
  await assert.rejects(async()=>{for await(const chunk of streamChat({...params,provider:{...provider,apiKey:padded},apiKey:padded})){}},(error:any)=>{noCredential(inspect(error));return error.status===401})
  assert.equal(sent.at(-1)!.trim(),key,'semantic Bearer token trimming remains protected')
  for(const empty of ['', ' ', '   ']){mode='normal';const chunks:any[]=[];for await(const chunk of streamChat({...params,provider:{...provider,apiKey:empty},apiKey:empty}))chunks.push(chunk);assert.equal(chunks.filter(chunk=>chunk.type==='content').map(chunk=>chunk.text).join(''),'Ordinary non-secret assistant text.','an absent/whitespace-only credential must not mask ordinary spaces')}
  mode='json-malformed'
  const longKey='FICTIONAL_ONLY_MALFORMED_RESPONSE_KEY_'.repeat(12)
  await assert.rejects(()=>quickComplete({...params,provider:{...provider,apiKey:longKey},apiKey:longKey}),(error:any)=>{assert.ok(!error.message.includes(longKey.slice(0,8)),'JSON parser previews must not leak a prefix after truncating a reflected credential');return /not valid bounded JSON/.test(error.message)})
  mode='json-abort'
  const abort=new AbortController(), timer=setTimeout(()=>abort.abort(),20)
  try {await assert.rejects(()=>quickComplete({...params,signal:abort.signal}),(error:any)=>error.name==='AbortError')} finally {clearTimeout(timer)}
  for(const selected of ['metadata','metadata-clean']){mode=selected;const chunks:any[]=[];for await(const chunk of streamChat(params))chunks.push(chunk);noCredential(JSON.stringify(chunks));const sample=chunks.find(chunk=>chunk.type==='usage');assert.ok(sample);assert.equal(sample.completionTokens,4);assert.equal(sample.promptTokens,9);assert.equal(sample.cachedTokens,5);assert.equal(sample.engine.decodeTokensPerSecond,15);assert.equal(sample.engine.decodeSeconds,2);assert.equal(sample.cortheon.searches,2);assert.equal(sample.cortheon.pagesRead,3);assert.equal(sample.cortheon.sources[0],'https://example.invalid/ordinary?q=green');assert.equal(sample.cortheon.sources.length,mode==='metadata'?1:2);if(mode==='metadata-clean')assert.equal(sample.cortheon.task,'Ordinary research')}
  mode='metadata-form';{const urlKey=`${key}~edge`;const chunks:any[]=[];for await(const chunk of streamChat({...params,provider:{...provider,apiKey:urlKey},apiKey:urlKey}))chunks.push(chunk);const sample=chunks.find(chunk=>chunk.type==='usage');assert.equal(sample.cortheon.sources.length,1,'a form-encoded source must be omitted when URLSearchParams would recover the actual sent key');assert.equal(sample.cortheon.sources[0],'https://example.invalid/ordinary?q=green');assert.equal(sample.cortheon.searches,2);assert.equal(sample.engine.decodeTokensPerSecond,15);assert.ok(!JSON.stringify(chunks).includes(urlKey))}
  mode='metadata-host';{const hostKey='fictional-host-token-5837';const chunks:any[]=[];for await(const chunk of streamChat({...params,provider:{...provider,apiKey:hostKey},apiKey:hostKey}))chunks.push(chunk);const sample=chunks.find(chunk=>chunk.type==='usage');assert.deepEqual(sample.cortheon.sources,['https://example.invalid/ordinary?q=green'],'canonical hostname decoding must not leave a credential-bearing source link');assert.equal(sample.cortheon.searches,2);assert.equal(sample.engine.decodeTokensPerSecond,15)}
  mode='finish-key';{const chunks:any[]=[];for await(const chunk of streamChat(params))chunks.push(chunk);noCredential(JSON.stringify(chunks))}
  for(const selected of ['tool-id-key','tool-name-key']){mode=selected;await assert.rejects(async()=>{for await(const chunk of streamChat(params))noCredential(JSON.stringify(chunk))},(error:any)=>{noCredential(inspect(error));return /tool identifiers/.test(error.message)})}
  mode='tool-args-positive';{const chunks:any[]=[];for await(const chunk of streamChat(params))chunks.push(chunk);assert.equal(chunks.find(chunk=>chunk.type==='tool_calls').toolCalls[0].function.arguments,JSON.stringify({value:key}),'ordinary successful tool argument semantics are unchanged')}
  let released=0;setEngineServeHook(async()=>({baseUrl:provider.baseUrl,apiKey:key,release:()=>{released++}} as any));mode='http';encoding=key
  await assert.rejects(async()=>{for await(const chunk of streamChat({...params,provider:{...provider,id:'tensorfold',source:'tensorfold',apiKey:'FICTIONAL_OLD_CALLER_KEY'},apiKey:'FICTIONAL_OLD_CALLER_KEY'})){}},(error:any)=>{noCredential(inspect(error));return error.status===401})
  assert.equal(sent.at(-1),key,'scrubbing follows the actual engine-routed outgoing credential')
  assert.equal(released,1)
  setEngineServeHook(null)

  const rendered:any[]=[],remote:any[]=[],observed:any[]=[];const observers=createObserverHub();observers.register('synthetic',event=>{observed.push(event)})
  store.win={isDestroyed:()=>false,webContents:{isDestroyed:()=>false,send:(_channel:string,payload:unknown)=>{rendered.push(payload)}}} as any;store.jobs=[];store.webhooks=[]
  const emitters=createEmitHelpers({remoteServer:()=>({publish:event=>{remote.push(event)}}),spineHost:()=>null,observerHub:()=>observers,persistMissions(){}})
  const thread:any={id:'synthetic-projection',title:'Synthetic task',kind:'chat',cwd:'',providerId:provider.id,model:'synthetic',status:'idle',createdAt:1,updatedAt:1,timeline:[],apiMessages:[{role:'user',content:'Synthetic fixed prompt.'}]};store.threads=[thread];mode='http';encoding=key
  await runAgent({thread,provider,model:'synthetic',temperature:0,reasoningEffort:'low',approvalMode:'readonly',autoCompact:false,maxSteps:1,maxWallMs:2000,signal:new AbortController().signal,persist(){},requestApproval:async()=>'deny',emit:event=>emitters.emitToRenderer(event)})
  for(const value of [rendered,remote,observed]){assert.ok(value.length>0);noCredential(JSON.stringify(value))}
  assert.match(JSON.stringify(remote),/HTTP 401/)
  rendered.length=0;remote.length=0;observed.length=0;mode='metadata'
  const metadataThread={...thread,id:'synthetic-metadata',timeline:[],apiMessages:[{role:'user',content:'Synthetic fixed metadata prompt.'}]};store.threads=[metadataThread]
  await runAgent({thread:metadataThread,provider,model:'synthetic',temperature:0,reasoningEffort:'low',approvalMode:'readonly',autoCompact:false,maxSteps:1,maxWallMs:2000,signal:new AbortController().signal,persist(){},requestApproval:async()=>'deny',emit:event=>emitters.emitToRenderer(event)})
  for(const value of [rendered,remote,observed])noCredential(JSON.stringify(value))
  assert.ok(JSON.stringify(remote).includes('https://example.invalid/ordinary?q=green'),'positive research metadata still reaches its configured projection')
  assert.ok(JSON.stringify(remote).includes('\"searches\":2'),'metadata counts are preserved in emitted turn statistics')
  console.log('Actual HTTP/SSE/JSON credential reflection, split streaming, attached headers, engine-routed key and runAgent renderer/remote/observer regression passed; normal text and status/reset/model diagnostics preserved.')
} finally {setEngineServeHook(null);Object.assign(store,saved);server.closeAllConnections();await new Promise<void>((resolve)=>server.close(()=>resolve()))}
