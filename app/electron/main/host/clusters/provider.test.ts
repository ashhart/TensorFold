import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { preserveManagedComputeProviders } from './provider'
import { classifyProvider, isManagedComputeProvider } from '../../../../shared/provider-trust'
import { assertProviderEndpoint } from '../../../agent/security'
const source=ts.createSourceFile('cluster-runtime.ts',readFileSync('electron/main/host/cluster-runtime.ts','utf8'),ts.ScriptTarget.ES2022,true)
let callback:ts.Expression|undefined
const walk=(node:ts.Node)=>{if(ts.isPropertyAssignment(node)&&node.name.getText(source)==='useEndpoint')callback=node.initializer;ts.forEachChild(node,walk)};walk(source);assert.ok(callback)
let active=true,fail=false,saved:any=null,events=0
const original:any={providers:[{id:'unrelated',name:'Existing',models:['keep'],apiKey:''}],unrelated:'preserve'},store:any={settings:original}
const environment:Record<string,unknown>={options:{serializeSettings:async(action:any)=>action()},current(){if(!active)throw new Error('Synthetic scope changed')},store,saveSettings:async(value:any,options:any)=>{options.validateCommit();if(fail)throw new Error('Synthetic provider write failed');saved=value},sendToRenderer(){events++},redactSettings:(value:any)=>value}
const code=ts.transpileModule('const promote='+ts.createPrinter().printNode(ts.EmitHint.Expression,callback,source)+';return promote;',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText
const promote=new Function(...Object.keys(environment),code)(...Object.values(environment))
const value={id:'synthetic-deployment',label:'Synthetic remote compute',baseUrl:'http://127.0.0.1:19000/v1',apiKey:'synthetic_inference_credential_'.repeat(3),model:'Example/Model'}
fail=true;await assert.rejects(()=>promote(value),/write failed/);assert.equal(store.settings,original);assert.equal(events,0);fail=false
await promote(value);const provider=saved.providers.find((row:any)=>row.id==='cluster:synthetic-deployment');assert.ok(provider);assert.equal(store.settings.unrelated,'preserve');assert.equal(provider.remoteCompute,true);assert.equal(provider.kind,'cloud');assert.equal(isManagedComputeProvider(provider),true);assert.equal(classifyProvider(provider),'external');assert.doesNotThrow(()=>assertProviderEndpoint(provider),'Verified loopback forwarding must be reachable without becoming trusted-local')
for(const patch of [{kind:'local'},{source:'tensorfold'},{remoteCompute:undefined},{transport:'codex-cli'}])assert.equal(classifyProvider({...provider,...patch}),'external','Managed identity cannot become local by tampering another display/transport field')
assert.throws(()=>assertProviderEndpoint({...provider,baseUrl:'http://192.168.1.4/v1'}),/HTTPS|verified/)
assert.throws(()=>assertProviderEndpoint({...provider,baseUrl:'https://169.254.169.254/v1'}),/metadata|link-local/)
assert.throws(()=>assertProviderEndpoint({...provider,baseUrl:'https://user:password@synthetic.example.invalid/v1'}),/embedded/)
assert.deepEqual(preserveManagedComputeProviders([{...provider,name:'Renamed'}],[provider])[0],{...provider,name:'Renamed'})
assert.deepEqual(preserveManagedComputeProviders([],[provider]),[],'Explicit removal remains available')
for(const patch of [{baseUrl:'http://127.0.0.1:19001/v1'},{apiKey:'replacement'},{models:['Other/Model']},{remoteCompute:undefined},{kind:'local'},{id:'other-provider'}])assert.throws(()=>preserveManagedComputeProviders([{...provider,...patch}],[provider]),/managed|Compute|connection|verified/)
assert.throws(()=>preserveManagedComputeProviders([provider],[]),/verified/)
const donated:any={...provider,id:'donation:synthetic-card',name:'Lent machine'}
assert.deepEqual(preserveManagedComputeProviders([{...donated,name:'Renamed'}],[donated])[0],{...donated,name:'Renamed'},'A donation provider may be renamed like a cluster endpoint')
assert.throws(()=>preserveManagedComputeProviders([{...donated,baseUrl:'http://127.0.0.1:19002/v1'}],[donated]),/Reconcile/,'A donation endpoint cannot be rewritten through a settings save')
assert.throws(()=>preserveManagedComputeProviders([{...donated,remoteCompute:undefined}],[]),/verified/,'A donation provider cannot be added by a settings save')
const reordered:any={source:provider.source,kind:provider.kind,remoteCompute:provider.remoteCompute,models:[...provider.models],apiKey:provider.apiKey,baseUrl:provider.baseUrl,auth:provider.auth,transport:provider.transport,id:provider.id,name:'Renamed'}
assert.deepEqual(preserveManagedComputeProviders([reordered],[provider])[0],{...provider,name:'Renamed'},'A settings round-trip may reorder fields and still be the same connection')
assert.deepEqual(preserveManagedComputeProviders([{...provider,name:'Renamed',spendBudget:null}],[provider])[0],{...provider,name:'Renamed'},'Unrelated metadata on the incoming row does not block a save')
active=false;await assert.rejects(()=>promote(value),/scope changed/)
console.log('Actual endpoint promotion callback and managed-provider policies: durable settings, remote trust through loopback, transport reachability, metadata/userinfo denial, renderer route/credential protection and rename/removal passed')
