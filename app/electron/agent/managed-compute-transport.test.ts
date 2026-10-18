import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { configureManagedComputeTransport, holdManagedComputeResponse } from './managed-compute-transport'
import { createProviderFetch } from './secure-fetch'
import { isManagedComputeProvider } from '../../shared/provider-trust'
import { providerWithAccount } from './provider-accounts'
import { isSecretMask, maskedSecretUrlError, SECRET_MASK } from './security'
const provider:any={id:'cluster:deployment-a',name:'Synthetic managed compute',kind:'cloud',source:'custom',remoteCompute:true,transport:'chat-completions',auth:'api-key',baseUrl:'http://127.0.0.1:19000/v1',apiKey:'synthetic_inference_credential_'.repeat(3),models:['Example/Model']}
let fallback=0
configureManagedComputeTransport(null)
await assert.rejects(()=>createProviderFetch(provider,async()=>{fallback++;return new Response('{}')})(provider.baseUrl+'/models'),/Reconnect/);assert.equal(fallback,0,'Managed requests never fall through to arbitrary stored-loopback fetch')
const file=ts.createSourceFile('cluster-runtime.ts',readFileSync('electron/main/host/cluster-runtime.ts','utf8'),ts.ScriptTarget.ES2022,true)
let callback:ts.Expression|undefined
const walk=(node:ts.Node)=>{if(ts.isCallExpression(node)&&node.expression.getText(file)==='configureManagedComputeTransport')callback=node.arguments[0];ts.forEachChild(node,walk)};walk(file);assert.ok(callback)
let role=true,profile='/synthetic/one',leases=0,calls=0,liveKey=provider.apiKey,controller!:ReadableStreamDefaultController<Uint8Array>
const sent:any[]=[],store:any={settings:{providers:[provider]}}
const host:any={id:'host-a',connection:{kind:'ssh',sshHostId:'ssh-a',port:8766}}
const runtime=async()=>({manager:{credentials:async()=>({baseUrl:'http://127.0.0.1:20000/v1',apiKey:liveKey,model:'Example/Model'}),options:{store:{get:async()=>({hosts:[host],deployments:[{id:'deployment-a',plan:{members:[{hostId:'host-a',rank:0}]}}]})}}},network:{consumer:async(_host:any,url:string,init:RequestInit)=>{calls++;sent.push({url,init});return new Response(new ReadableStream<Uint8Array>({start(value){controller=value}}),{headers:{'content-type':'text/event-stream'}})}}})
const environment:Record<string,unknown>={vontraHomeDir:()=>profile,store,requireSecurityRole(){if(!role)throw new Error('Synthetic role revoked')},isManagedComputeProvider,holdManagedComputeResponse,runtime,options:{workspaceMutationGuard:{reserveStart(){leases++;let released=false;return{release(){if(!released){released=true;leases--}}}}}},resolveRemoteOrchestratorRequest(){throw new Error('SSH inference must not use DNS/local-port fallback')},createPinnedFetch(){throw new Error('SSH inference must not use ordinary fetch')}}
const code=ts.transpileModule('const bridge='+ts.createPrinter().printNode(ts.EmitHint.Expression,callback,file)+';return bridge;',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText
const bridge=new Function(...Object.keys(environment),code)(...Object.values(environment))
configureManagedComputeTransport(bridge)
for(const url of ['http://127.0.0.1:19001/v1/chat/completions',provider.baseUrl+'/responses?key=forbidden',provider.baseUrl+'/../api/cluster/v1/status'])await assert.rejects(()=>createProviderFetch(provider)(url,{method:'POST',body:'{}'}),/Unsupported/)
assert.equal(calls,0)
liveKey='synthetic_changed_credential_'.repeat(3);await assert.rejects(()=>createProviderFetch(provider)(provider.baseUrl+'/chat/completions',{method:'POST',body:'{}'}),/credential changed/);assert.equal(calls,0);liveKey=provider.apiKey
role=false;await assert.rejects(()=>createProviderFetch(provider)(provider.baseUrl+'/models'),/revoked/);assert.equal(calls,0);role=true
const response=await createProviderFetch(provider,async()=>{fallback++;return new Response('{}')})(provider.baseUrl+'/chat/completions',{method:'POST',body:'{}'})
assert.equal(sent[0].url,'http://127.0.0.1:20000/v1/chat/completions','Reviewed deployment identity resolves a fresh owned SSH route rather than dispatching to the old stored port');assert.equal(new Headers(sent[0].init.headers).get('Authorization'),'Bearer '+provider.apiKey);assert.equal(fallback,0)
assert.equal(leases,1,'The operation lease must remain held after headers while the inference response stream is still active')
controller.enqueue(new TextEncoder().encode('data: synthetic\n\n'));controller.close();assert.match(await response.text(),/synthetic/);assert.equal(leases,0,'End of response physically releases the inference lease')
const abandoned=await createProviderFetch(provider)(provider.baseUrl+'/chat/completions',{method:'POST',body:'{}'});assert.equal(leases,1);await abandoned.body!.cancel('Synthetic explicit Stop');assert.equal(leases,0,'Explicit cancellation releases its reservation only after source cancellation')
const probeFile=ts.createSourceFile('provider-runtime.ts',readFileSync('electron/main/host/provider-runtime.ts','utf8'),ts.ScriptTarget.ES2022,true)
const probeFunction=probeFile.statements.find((node):node is ts.FunctionDeclaration=>ts.isFunctionDeclaration(node)&&node.name?.text==='resolveProviderProbe');assert.ok(probeFunction)
const probeCode=ts.transpileModule(ts.createPrinter().printNode(ts.EmitHint.Unspecified,probeFunction,probeFile).replace(/^export /,'')+'\nreturn resolveProviderProbe;',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText
const resolveProbe=new Function('store','providerWithAccount','isSecretMask','maskedSecretUrlError',probeCode)(store,providerWithAccount,isSecretMask,maskedSecretUrlError)
const probeProvider=resolveProbe({...provider,apiKey:SECRET_MASK});assert.deepEqual(probeProvider.cliEnv,{});assert.equal(probeProvider.apiKey,provider.apiKey)
const modelProbe=await createProviderFetch(probeProvider)(provider.baseUrl+'/models');assert.equal(sent.at(-1).url,'http://127.0.0.1:20000/v1/models');assert.equal(leases,1);controller.enqueue(new TextEncoder().encode('{"data":[{"id":"Example/Model"}]}'));controller.close();assert.deepEqual(await modelProbe.json(),{data:[{id:'Example/Model'}]});assert.equal(leases,0,'The actual resolved model-probe shape keeps the current route and releases after consumption')
configureManagedComputeTransport(null)
let releases=0,cancelCalls=0,physicalCancel!:()=>void
const physicalSource=new ReadableStream<Uint8Array>({cancel(){cancelCalls++;return new Promise<void>(resolve=>{physicalCancel=resolve})}})
const held=holdManagedComputeResponse(new Response(physicalSource),()=>{releases++})
const cancelling=held.body!.cancel();await Promise.resolve();assert.equal(cancelCalls,1);assert.equal(releases,0,'A pending physical cancellation still owns its profile lease');physicalCancel();await cancelling;assert.equal(releases,1)
let failedController!:ReadableStreamDefaultController<Uint8Array>
const failed=holdManagedComputeResponse(new Response(new ReadableStream<Uint8Array>({start(value){failedController=value}})),()=>{releases++})
const readFailure=failed.text();failedController.error(new Error('Synthetic stream failure'));await assert.rejects(()=>readFailure,/stream failure/);assert.equal(releases,2)
let oversizedCancelled=0
const oversized=holdManagedComputeResponse(new Response(new ReadableStream<Uint8Array>({start(value){value.enqueue(new Uint8Array(9))},cancel(){oversizedCancelled++}})),()=>{releases++},8)
await assert.rejects(()=>oversized.text(),/size limit/);assert.equal(oversizedCancelled,1);assert.equal(releases,3)
const empty=holdManagedComputeResponse(new Response(null,{status:204}),()=>{releases++});assert.equal(empty.status,204);assert.equal(releases,4)
let unreadController!:ReadableStreamDefaultController<Uint8Array>,unreadReleases=0
const unread=holdManagedComputeResponse(new Response(new ReadableStream<Uint8Array>({start(value){unreadController=value;value.enqueue(new Uint8Array([1]))}})),()=>{unreadReleases++})
await new Promise<void>(resolve=>setImmediate(resolve));assert.equal(unreadController.desiredSize,1,'The wrapper has prefetched the first source chunk while its caller is not reading');assert.equal(unreadReleases,0)
unreadController.error(new Error('Synthetic unread response failed physically'));await new Promise<void>(resolve=>setImmediate(resolve));assert.equal(unreadReleases,1,'Physical source failure releases the lease despite output backpressure');await assert.rejects(()=>unread.text(),/unread response failed physically/);await unread.body!.cancel().catch(()=>{});assert.equal(unreadReleases,1,'Later cleanup must not double release a physically failed source')
console.log('Actual managed compute bridge and stream reservation: default closed, fresh exact route/key, legitimate model probe, strict routes, no local-port fallback, complete/cancel/error/backpressure/size-limit physical settlement passed')
