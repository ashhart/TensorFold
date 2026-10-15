import { mergeThreadEventPatch } from '../../../src/lib/store/shared/merge'
import { mergeThreadRecord } from '../../../src/lib/store/shared/merge'
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import type { AgentEvent, Thread } from '../../agent/types'
import { projectedHostId, projectionOriginsMatch, queuePendingAgentEvent, drainPendingAgentEvents, clearPendingAgentEvents, quarantineProjectionCollision } from '../../../src/lib/store/shared/proxy-state'
import { threadStatusFromRun } from '../../../src/lib/store/types'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-memory-publication-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { createThreadMemoryPolicyHost } = await import('./thread-memory-policy-host')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const oldFetch = globalThis.fetch, rename = fs.rename
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, jobs: store.jobs, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, skills: store.skills, suggestions: store.suggestions }
const provider = { id: 'synthetic-memory-publication', name: 'Synthetic API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9896/v1', apiKey: '', models: ['synthetic'] }
let modelRequests = 0, holdModel: (() => Promise<void>) | undefined
const sse = () => new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic generation finished.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'all provider requests use the inert fixture')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  modelRequests++
  if (holdModel) await holdModel()
  return sse()
}) as typeof fetch

// Execute the production renderer reducer, not a duplicate patch implementation.
// Its collaborators below are only origin/status helpers and an in-memory store.
const rendererSource = await fs.readFile(path.join(process.cwd(), 'src/lib/store/slices/threadsSlice.ts'), 'utf8')
const tree = ts.createSourceFile('threadsSlice.ts', rendererSource, ts.ScriptTarget.Latest, true)
let applySource = ''
function visit(node: ts.Node) { if (ts.isMethodDeclaration(node) && node.name.getText(tree) === 'applyEvent') applySource = node.getText(tree); ts.forEachChild(node, visit) }
visit(tree); assert.ok(applySource)
const reducerJs = ts.transpileModule(`function ${applySource}`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const makeReducer = new Function('get', 'set', 'projectedHostId', 'projectionOriginsMatch', 'threadStatusFromRun', 'mergeThreadEventPatch', 'queuePendingAgentEvent', 'quarantineProjectionCollision', `${reducerJs}\nreturn applyEvent;`)
let updateSource=''
const scanUpdate=(node:any)=>{if(ts.isMethodDeclaration(node)&&node.name.getText(tree)==='applyThreadUpdate')updateSource=node.getText(tree);ts.forEachChild(node,scanUpdate)};scanUpdate(tree)
const updateJs=ts.transpileModule(`function ${updateSource}`,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText
const makeUpdate=new Function('get','set','projectedHostId','projectionOriginsMatch','quarantineProjectionCollision','clearPendingAgentEvents','mergeThreadRecord','drainPendingAgentEvents',`${updateJs}\nreturn applyThreadUpdate;`)

async function waitFor(predicate: () => boolean, label: string) { for (let i = 0; i < 800; i++) { if (predicate()) return; await new Promise((r) => setTimeout(r, 5)) }; throw new Error(`Timed out: ${label}`) }
async function fixture(id: string, extra: Partial<Thread> = {}) {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', approvalMode: 'readonly', skillSelfImprovement: false, autoCompact: false }
  store.projects = []; store.jobs = []; store.projectMemory = []; store.projectMemoryCandidates = []; store.skills = []; store.suggestions = []
  const thread: Thread = { id, title: 'Synthetic retained chat', kind: 'chat', cwd: '/synthetic-workspace', workspace: { kind: 'ssh', hostId: 'inert-workspace', path: '/synthetic-workspace' }, providerId: provider.id, providerAccountId: null, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'SYNTHETIC-RETAINED-HISTORY' }], memoryPolicy: { useExisting: true, contribute: true }, memoryPolicyRevision: 0, ...extra }
  store.threads = [thread]
  if (!thread.incognito) await storage.saveThreads(store.threads)
  let rendered: any = { tickets: [], threads: [structuredClone(thread)], status: {}, approvals: {}, view: { name: 'thread', id } }
  const apply = makeReducer(() => rendered, (patch: any) => { rendered = { ...rendered, ...(typeof patch === 'function' ? patch(rendered) : patch) } }, projectedHostId, projectionOriginsMatch, threadStatusFromRun, mergeThreadEventPatch, queuePendingAgentEvent, quarantineProjectionCollision)
  const update=makeUpdate(()=>({...rendered,applyEvent:apply}),(patch:any)=>{rendered={...rendered,...(typeof patch==='function'?patch(rendered):patch)}},projectedHostId,projectionOriginsMatch,quarantineProjectionCollision,clearPendingAgentEvents,mergeThreadRecord,drainPendingAgentEvents)
  const events: AgentEvent[] = [], rawMemoryAliases: boolean[] = []
  const publish = (event: AgentEvent) => {
    if (event.type === 'thread_update' && event.patch.memoryPolicyRevision !== undefined) rawMemoryAliases.push(event.patch.memoryPolicy === thread.memoryPolicy)
    const wire = structuredClone(event); events.push(wire); apply(wire)
  }
  const aborters = new Map<string, AbortController>(), guard = new WorkspaceMutationGuard()
  let settled = 0
  const runtime = createThreadRuntime({ aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer: publish, persistThreads() {},
    fullToolContext: (run) => ({ cwd: '/synthetic-workspace', backend: { kind: 'ssh' }, currentThread: thread, memoryPolicy: run?.memoryPolicy }), approvalResolver: () => async () => 'allow', isHighRiskTool: () => false,
    claimJob: () => ({ id: `job-${id}`, status: 'running', metadata: {}, pendingApprovals: [] } as any), patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob() { settled++ }, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 128000,
    captureCheckpoint: async () => ({ ok: true, sha: 'synthetic' }), recordTurnCheckpoint() {},
  })
  const change = createThreadMemoryPolicyHost({ threads: () => store.threads, provider: () => provider, aborters, guard, publish: (accepted) => publish({ type: 'thread_update', runId: '', threadId: id, patch: { memoryPolicy: { ...accepted.memoryPolicy! }, memoryPolicyRevision: accepted.memoryPolicyRevision } }) })
  return { thread, runtime, aborters, guard, change, events, rawMemoryAliases, rendered: () => rendered.threads[0] as Thread, settled: () => settled,
    apply,forgetThread:()=>{rendered.threads=[]},hydrate:(t:any)=>update(t),memoryEvents: () => events.filter((event) => event.type === 'thread_update' && event.patch.memoryPolicyRevision !== undefined) }
}
try {
 const f=await fixture('owned-pending-replay')
 f.forgetThread()
 await f.runtime.startThreadRun({threadId:f.thread.id,userText:'One synthetic ordinary completion while renderer has not hydrated its thread.'})
 await waitFor(()=>f.settled()===1&&!f.aborters.size,'owned run settled')
 assert.equal(f.thread.memoryPolicyRevision,1)
 const hostChange=createThreadMemoryPolicyHost({threads:()=>store.threads,provider:()=>provider,aborters:f.aborters,guard:f.guard,publish:accepted=>f.hydrate(structuredClone(accepted))})
 await hostChange(f.thread.id,{useExisting:false,contribute:false},1)
 assert.equal(f.thread.memoryPolicyRevision,2)
 assert.deepEqual(f.thread.memoryPolicy,{useExisting:false,contribute:false})
 await assert.doesNotReject(()=>hostChange(f.thread.id,{useExisting:true,contribute:false},f.rendered().memoryPolicyRevision??0),'draining older pending startup events must not roll back a newer hydrated memory revision')
 assert.equal(f.rendered().memoryPolicyRevision,3)
 assert.deepEqual(f.rendered().memoryPolicy,{useExisting:true,contribute:false})
 assert.ok(f.rendered().timeline.some(item=>item.text==='Synthetic generation finished.'))
 const beforeReplayHistory=structuredClone(f.rendered().timeline)
 const currentPair={memoryPolicy:{...f.rendered().memoryPolicy!},memoryPolicyRevision:f.rendered().memoryPolicyRevision}
 f.apply({type:'thread_update',threadId:f.thread.id,runId:'older',patch:{memoryPolicy:{useExisting:true,contribute:true},memoryPolicyRevision:1,model:'synthetic-unrelated-patch'}})
 assert.equal(f.rendered().memoryPolicyRevision,currentPair.memoryPolicyRevision)
 assert.deepEqual(f.rendered().memoryPolicy,currentPair.memoryPolicy)
 assert.equal(f.rendered().model,'synthetic-unrelated-patch','stale memory metadata does not discard unrelated host patch fields')
 assert.deepEqual(f.rendered().timeline,beforeReplayHistory)
 f.apply({type:'thread_update',threadId:f.thread.id,orchestratorHostId:'synthetic-foreign-origin',runId:'foreign',patch:{memoryPolicy:{useExisting:false,contribute:true},memoryPolicyRevision:90}})
 assert.equal(f.rendered().memoryPolicyRevision,currentPair.memoryPolicyRevision,'another origin cannot replace the local memory pair')
 f.apply({type:'thread_update',threadId:'synthetic-other-chat',runId:'other',patch:{memoryPolicy:{useExisting:false,contribute:true},memoryPolicyRevision:90}})
 assert.equal(f.rendered().memoryPolicyRevision,currentPair.memoryPolicyRevision,'another thread cannot replace this chat memory pair')
 f.apply({type:'thread_update',threadId:f.thread.id,runId:'damaged',patch:{memoryPolicy:{useExisting:false,contribute:false},memoryPolicyRevision:'damaged'}} as any)
 assert.equal(f.rendered().memoryPolicyRevision,'damaged','a malformed present incoming revision remains visible')
 f.apply({type:'thread_update',threadId:f.thread.id,runId:'older-again',patch:{memoryPolicy:{useExisting:true,contribute:true},memoryPolicyRevision:1,model:'synthetic-after-damage'}})
 assert.equal(f.rendered().memoryPolicyRevision,'damaged','an older queued event cannot make unknown saved metadata look repaired')
 assert.equal(f.rendered().model,'synthetic-after-damage')
 f.hydrate({...structuredClone(f.thread),updatedAt:Date.now()+10})
 assert.equal(f.rendered().memoryPolicyRevision,3,'a fresh full host snapshot can reconcile renderer-only damaged state')
 f.apply({type:'thread_update',threadId:f.thread.id,runId:'damaged-policy',patch:{memoryPolicy:null,memoryPolicyRevision:4}} as any)
 assert.equal(f.rendered().memoryPolicy,null,'damaged present policy does not become legacy defaults')
 f.apply({type:'thread_update',threadId:f.thread.id,runId:'older-policy',patch:{memoryPolicy:{useExisting:true,contribute:true},memoryPolicyRevision:1}})
 assert.equal(f.rendered().memoryPolicy,null,'an older valid event cannot hide a newer malformed policy')
 assert.equal(f.rendered().memoryPolicyRevision,4)
 clearPendingAgentEvents('synthetic-other-chat')

 console.log('Actual startup/protected-store/pending-event hydration/renderer/host CAS preserves the latest accepted memory pair and streamed history')
}finally{fs.rename=rename;holdModel=undefined;globalThis.fetch=oldFetch;await new Promise(r=>setTimeout(r,100));while(storage.writeChains.size)await Promise.allSettled([...storage.writeChains.values()]);Object.assign(store,saved);storage.closeSessionSearchIndexes();await fs.rm(home,{recursive:true,force:true,maxRetries:10,retryDelay:20})}
