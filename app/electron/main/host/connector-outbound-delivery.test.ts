import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { personalMessageDraftEvent } from '../../agent/tool-labels'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-reviewed-message-'))
const key = '__reviewedMessageTest'
const fixtureState = { handlers: new Map<string, Function>(), store: { connectorEvents: [] as any[], projects: [], settings: {whatsapp:{sources:[{id:'synthetic-account',enabled:true,paired:true,phoneNumber:'15550000001'}]}} }, writes: [] as any[], failWrite: 0, writeCount: 0, afterWrite: null as null | (()=>void | Promise<void>) }
;(globalThis as any)[key] = fixtureState
try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, `export const store=globalThis.${key}.store;export const handle=(c,f)=>globalThis.${key}.handlers.set(c,f);export const persistConnectorEvents=()=>{};export const persistProjectMemoryCandidates=()=>{};export const saveConnectorEvents=async(events)=>{const state=globalThis.${key};state.writeCount++;if(state.failWrite===state.writeCount)throw new Error('Synthetic durable storage failure');state.writes.push(JSON.parse(JSON.stringify(events)));await state.afterWrite?.();};export const requireSecurityRole=()=>{};export const auditActor=()=>'synthetic-owner';export const auditEvent=async()=>{};`)
  const noop = path.join(tmp, 'noop.mjs')
  await fs.writeFile(noop, ['Email','IMessage','WhatsApp','Discord','Mattermost','Matrix','GoogleChat','Slack','Calendar','Document','Browser'].map(n=>`export const register${n}Ipc=()=>{};`).join('\n'))
  const alias: Record<string, string> = { './ipc': fixture, './persistence': fixture, './audit': fixture, '../state': fixture, '../../store': fixture }
  for (const n of ['email','imessage','whatsapp','discord','mattermost','matrix','googlechat','slack','calendar','documents','browser']) alias[`./connector-ipc/${n}`] = noop
  const jiti = createJiti(__filename, { alias, fsCache: false, moduleCache: false })
  const { registerAllConnectorIpc } = await jiti.import<any>(path.join(__dirname, 'connector-ipc.ts'))
  const sends: any[] = []
  let transportFailure = false
  let onTransport: null | (()=>void | Promise<void>) = null
  registerAllConnectorIpc({
    spine: { rememberConnectorEvents: async()=>{}, recordEventDelivery: async (event:any, delivery:any)=>{ event.delivery=delivery }, persistDecision: async()=>{} },
    emailTransport: { isEmailSendRequest:()=>false, transportApprovedEmailEvent:async()=>{} },
    channels: { resolveChannelApprovalByThread:()=>{}, channelAdapters:[{platform:'whatsapp', send:async(address:any, reply:any)=>{ sends.push({address,reply}); await onTransport?.(); if(transportFailure)throw new Error('Synthetic timeout after request admission'); return {ok:true,sentId:'synthetic-message-1'} }}] },
    callbacks: { rememberApprovalRule:async()=>({id:'old-broad-policy'}) },
  })
  const { sealConnectorOutboundEvent, recoverInterruptedConnectorDeliveries } = await jiti.import<any>(path.join(__dirname, 'connector-outbound-delivery.ts'))
  const event = sealConnectorOutboundEvent(personalMessageDraftEvent({ connectorId:'whatsapp', target:'15550000000@lid', body:'Exact reviewed body.', threadKey:'whatsapp:synthetic-account:15550000000@lid' }))
  fixtureState.store.connectorEvents=[event]
  await fixtureState.handlers.get('assistant:resolveApproval')!({}, { eventId:event.id, decision:'approve', reviewDigest:(event as any).reviewDigest })
  assert.equal(sends.length,1,'approved personal message must dispatch through the actual registered host approval path')
  assert.equal(sends[0].address.account,'synthetic-account')
  assert.equal(sends[0].reply.text,'Exact reviewed body.')
  assert.equal(fixtureState.store.connectorEvents[0]?.delivery?.status,'delivered','approval is distinct from confirmed dispatch')
  assert.ok(fixtureState.writes.some(events=>events[0]?.delivery?.status==='sending'),'durable one-shot marker precedes transport')
  const resolve = fixtureState.handlers.get('assistant:resolveApproval')!
  const reset = (body='Exact reviewed body.') => {
    fixtureState.writes=[];fixtureState.failWrite=0;fixtureState.writeCount=0;fixtureState.afterWrite=null;sends.length=0;transportFailure=false;onTransport=null
    fixtureState.store.settings.whatsapp.sources[0].phoneNumber='15550000001'
    const next = sealConnectorOutboundEvent(personalMessageDraftEvent({connectorId:'whatsapp',target:'15550000000@lid',body,accountId:'synthetic-account'}))
    fixtureState.store.connectorEvents=[next]
    return next
  }
  const approve = (next:any,digest:unknown=next.reviewDigest) => resolve({}, {eventId:next.id,decision:'approve',reviewDigest:digest})
  const missing = reset()
  await assert.rejects(()=>approve(missing,null),/not fully reviewed|changed/i)
  assert.equal(sends.length,0,'outbound approval requires the exact reviewed host digest')
  const edited = reset()
  edited.payload.body='Changed after review'
  await assert.rejects(()=>approve(edited),/not fully reviewed|changed/i)
  assert.equal(sends.length,0,'body edits cannot inherit an earlier one-time approval')
  const changedScope = reset()
  fixtureState.store.settings.whatsapp.sources[0].phoneNumber='15550000002'
  await assert.rejects(()=>approve(changedScope),/not fully reviewed|changed/i)
  assert.equal(sends.length,0,'account identity is bound to the reviewed snapshot')

  const storage = reset()
  fixtureState.failWrite=1
  await assert.rejects(()=>approve(storage),/durable storage failure/)
  assert.equal(sends.length,0,'transport cannot precede a durable one-shot marker')
  assert.equal(fixtureState.store.connectorEvents[0].approvalStatus,'pending')
  fixtureState.failWrite=0
  await approve(storage)
  assert.equal(sends.length,1,'a pre-admission storage failure remains safely retryable')

  const stale = reset()
  fixtureState.afterWrite=()=>{if(fixtureState.writeCount===1)fixtureState.store.connectorEvents[0].payload.body='Mutation during durable approval await'}
  await assert.rejects(()=>approve(stale),/changed before dispatch/)
  assert.equal(sends.length,0,'live body mutation during durable approval is refused before adapter admission')
  assert.equal(fixtureState.store.connectorEvents[0].delivery.status,'failed')
  const rotated = reset()
  fixtureState.afterWrite=()=>{if(fixtureState.writeCount===1)fixtureState.store.settings.whatsapp.sources[0].phoneNumber='15550000002'}
  await assert.rejects(()=>approve(rotated),/changed before dispatch/)
  assert.equal(sends.length,0,'account rotation during the protected write cannot change the sender')
  const removed = reset()
  fixtureState.afterWrite=()=>{if(fixtureState.writeCount===1)fixtureState.store.connectorEvents=[]}
  await assert.rejects(()=>approve(removed),/changed before dispatch/)
  assert.equal(sends.length,0,'evicted or replaced approvals never send after an await')

  const timeout = reset()
  transportFailure=true
  const unknown = await approve(timeout)
  assert.equal(unknown.delivery.status,'uncertain')
  await approve(timeout)
  assert.equal(sends.length,1,'unknown delivery completion is not automatically retried by another approval')

  const finalSave = reset()
  fixtureState.failWrite=2
  await assert.rejects(()=>approve(finalSave),/durable storage failure/)
  assert.equal(sends.length,1)
  const persistedBeforeCrash=fixtureState.writes[0]
  assert.equal(persistedBeforeCrash[0].delivery.status,'sending')
  fixtureState.failWrite=0;fixtureState.afterWrite=null
  fixtureState.store.connectorEvents=structuredClone(persistedBeforeCrash)
  await recoverInterruptedConnectorDeliveries()
  assert.equal(fixtureState.store.connectorEvents[0].delivery.status,'uncertain','restart cannot claim a recipient delivery whose receipt was never committed')
  await approve(finalSave)
  assert.equal(sends.length,1,'restart preserves the consumed one-shot attempt and refuses replay')

  ;(fixtureState.store.settings as any).imessage={enabled:true,allowedChatIds:'allowed@example.test'}
  const outside=sealConnectorOutboundEvent(personalMessageDraftEvent({connectorId:'imessage',target:'blocked@example.test',body:'Synthetic review'}))
  assert.match(outside.outboundError,/current connection scope/,'model-composed iMessage cannot leave the current configured contact scope')
  const exactBody='  '+('words 🌿 '.repeat(300))+'  '
  const long = reset(exactBody)
  await approve(long)
  assert.ok(sends.length>1,'bounded transport chunks are used for long reviewed content')
  assert.equal(sends.map(send=>send.reply.text).join(''),exactBody,'every reviewed character survives the existing channel routing/split path')
  assert.ok(fixtureState.writes[0][0].outbound.body===exactBody,'durable attempt stores the same complete reviewed body')
  console.log('reviewed outbound registered-host, exact body/account binding, durable admission, storage failures, mutation races, timeout, restart and non-replay regressions passed')
} finally { delete (globalThis as any)[key]; await fs.rm(tmp,{recursive:true,force:true}) }
