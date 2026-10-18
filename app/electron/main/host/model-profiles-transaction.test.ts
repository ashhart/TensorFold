import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import type { Settings } from '../../agent/types'
import { builtinEngineProvider } from '../../agent/engine/provider'

const app=path.resolve(__dirname,'../../..')
const requireApp=createRequire(path.join(app,'package.json'))
const ts=requireApp('typescript')
const {makeHarness,seedModel}=await import('../../agent/engine/test-fixtures/harness')
const h=await makeHarness()
process.env.VONTRA_HOME=h.root
process.env.VONTRA_AUDIT_ACTOR='synthetic-profile-review'
const {saveSettings,loadSettings,closeSessionSearchIndexes,vontraHomeDir}=await import('../../store')
const {defaultSettings}=await import('../../store/defaults')
const {createSerialTaskQueue}=await import('../../agent/serial-task-queue')
const {redactSettings}=await import('../../agent/security')
const {requireSecurityRole,flushAuditEvents}=await import('./audit')
const {WorkspaceMutationGuard}=await import('./workspace-mutation-guard')
const profiles=await import('../../../shared/model-profiles')
const {EngineService}=await import('../../agent/engine/service')
const source=await fs.readFile(path.join(app,'electron/main/host/core-ipc.ts'),'utf8')
const ast=ts.createSourceFile('core-ipc.ts',source,ts.ScriptTarget.ES2022,true)
let commit:any,handler:any
function visit(node:any) {
 if(ts.isFunctionDeclaration(node)&&node.name?.text==='commitSettingsChange')commit=node
 if(ts.isCallExpression(node)&&node.expression.getText(ast)==='handle'&&node.arguments[0]?.getText(ast)==="'engine:profilesSave'")handler=node.arguments[1]
 ts.forEachChild(node,visit)
}
visit(ast);assert.ok(commit);assert.ok(handler)
const extracted=`${commit.getText(ast)}\nreturn ${handler.getText(ast)};`
const compiled=ts.transpileModule(extracted,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText
const provider={...builtinEngineProvider(),models:['Fixture/A','Fixture/B']}
const store: { settings: Settings }={settings:{...defaultSettings(),providers:[provider],activeProviderId:provider.id,activeModel:'Fixture/A',engineModelProfiles:[]}}
const base={contextTokens:8192,maxOutputTokens:1024,temperature:0.1,topP:1,topK:0,minP:0,speculativeDecoding:'auto' as const}
let diskFail=false, engineFailOnLarge=false
const events:any[]=[]
const service=new EngineService({binaryPath:'/synthetic',binary:h.binary,appHome:h.root,childEnv:h.env,pollMs:15,readyTimeoutMs:3000,totalMemoryBytes:32*1024**3,emit:()=>{},modelSettings:(model:string)=>{
 const value=profiles.activeEngineModelSettings(store.settings,model)
 if(engineFailOnLarge&&value?.contextTokens===16384)h.env.FAKE_SERVE_FAIL='Synthetic large-profile engine refused'
 else delete h.env.FAKE_SERVE_FAIL
 return value
}})
const injected={store,workspaceMutationGuard:new WorkspaceMutationGuard(),settingsChanges:createSerialTaskQueue(),
 saveSettings:async(next:any,options:Parameters<typeof saveSettings>[1])=>{if(diskFail)throw new Error('Synthetic disk unavailable');await saveSettings(next,options)},redactSettings,auditPreview:(value:any)=>String(value),vontraHomeDir,
 requireSecurityRole,validateEngineModelProfileSet:profiles.validateEngineModelProfileSet,assertEngineModelSamplingSupported:profiles.assertEngineModelSamplingSupported,
 updateEngineModelProfiles:profiles.updateEngineModelProfiles,engineService:()=>service,sendToRenderer:(...args:any[])=>events.push(args)}
const invoke=new Function(...Object.keys(injected),compiled)(...Object.values(injected))
const input=(model:string,expectedRevision:number,contextTokens=8192)=>({model,expectedRevision,activeProfileId:'default',profiles:[{id:'default',name:'Default',...base,contextTokens}]})
let checks=0
const check=async(name:string,run:()=>Promise<void>)=>{await run();checks++;console.log(`PASS ${name}`)}
try {
 await seedModel(h.hub,'Fixture/A');await seedModel(h.hub,'Fixture/B')
 await saveSettings(store.settings)
 await check('owner role required before model or disk operations',async()=>{
  process.env.VONTRA_SECURITY_ROLE='operator'
  await assert.rejects(()=>invoke({},input('Fixture/A',0)),/requires owner/)
  assert.deepEqual(store.settings.engineModelProfiles,[])
  process.env.VONTRA_SECURITY_ROLE='owner'
 })
 await check('malformed input and uninstalled arbitrary path rejected without settings mutation',async()=>{
  await assert.rejects(()=>invoke({},input('/private/forbidden',0)),/no longer installed/)
  await assert.rejects(()=>invoke({},input('Fixture/A',0,999999)),/at most/)
  await assert.rejects(()=>invoke({},{...input('Fixture/A',0),profiles:[{...base,id:'bad',name:'Bad',temperature:NaN}]}),/Temperature/)
  assert.deepEqual(store.settings.engineModelProfiles,[])
 })
 await check('protected persistence failure leaves live configuration and engine untouched',async()=>{
  diskFail=true
  await assert.rejects(()=>invoke({},input('Fixture/A',0)),/disk unavailable/)
  diskFail=false
  assert.deepEqual(store.settings.engineModelProfiles,[])
  assert.deepEqual((await loadSettings()).engineModelProfiles,[])
  assert.equal(service.supervisor.status().state,'stopped')
 })
 await check('same-model concurrent expected revision has one durable winner',async()=>{
  const results=await Promise.allSettled([invoke({},input('Fixture/A',0)),invoke({},input('Fixture/A',0))])
  assert.deepEqual(results.map((x:any)=>x.status).sort(),['fulfilled','rejected'])
  assert.equal(store.settings.engineModelProfiles![0].revision,1)
  assert.equal((await loadSettings()).engineModelProfiles?.[0].revision,1)
 })
 await service.load('Fixture/A')
 const lease=await service.supervisor.acquire('Fixture/A',{start:false})
 const pid=(await h.records()).filter((r:any)=>r.kind==='serve').at(-1).pid
 await check('active same-model request rejects apply before any durable mutation',async()=>{
  await assert.rejects(()=>invoke({},input('Fixture/A',1,16384)),/active.*requests/)
  assert.equal(store.settings.engineModelProfiles![0].revision,1)
  assert.equal((await loadSettings()).engineModelProfiles?.[0].revision,1)
  assert.equal(service.supervisor.activeRequestCount,1)
 })
 await check('idle other-model profile save preserves active request and process',async()=>{
  await invoke({},input('Fixture/B',0))
  assert.equal(service.supervisor.activeRequestCount,1)
  assert.equal(service.supervisor.live()?.model,'Fixture/A')
  assert.equal((await h.records()).filter((r:any)=>r.kind==='serve').at(-1).pid,pid)
  assert.equal(store.settings.engineModelProfiles!.find((row:any)=>row.model==='Fixture/B')!.revision,1)
 })
 lease.release()
 await check('failed restart rolls back durable profile and restores previous working engine',async()=>{
  engineFailOnLarge=true
  await assert.rejects(()=>invoke({},input('Fixture/A',1,16384)),/Synthetic large-profile engine refused/)
  assert.equal(profiles.activeEngineModelSettings(store.settings,'Fixture/A')?.contextTokens,8192)
  const persisted=await loadSettings()
  assert.equal(profiles.activeEngineModelSettings(persisted,'Fixture/A')?.contextTokens,8192)
  assert.equal(store.settings.engineModelProfiles!.find((row:any)=>row.model==='Fixture/A')!.revision,1)
  assert.equal(store.settings.engineModelProfiles!.find((row:any)=>row.model==='Fixture/B')!.revision,1)
  assert.equal(service.supervisor.status().state,'ready')
  assert.equal(service.supervisor.live()?.model,'Fixture/A')
  engineFailOnLarge=false
 })
 await check('idle loaded profile edit reaches real launch arguments and durable reload',async()=>{
  await invoke({},input('Fixture/A',1,16384))
  const record=(await h.records()).filter((r:any)=>r.kind==='serve').at(-1)
  const index=record.args.indexOf('--context')
  assert.equal(record.args[index+1],'16384')
  assert.equal(profiles.activeEngineModelSettings(await loadSettings(),'Fixture/A')?.contextTokens,16384)
  assert.equal(store.settings.engineModelProfiles!.find((row:any)=>row.model==='Fixture/A')!.revision,2)
 })
 console.log(`Model profile host transactions passed ${checks} cases against actual protected storage and engine supervisor`)
} finally {
 process.env.VONTRA_SECURITY_ROLE='owner'
 await service.unload().catch(()=>{});service.shutdown();await flushAuditEvents();closeSessionSearchIndexes();await h.cleanup()
}
