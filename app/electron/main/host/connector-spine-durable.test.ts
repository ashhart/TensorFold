import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { personalMessageDraftEvent } from '../../agent/tool-labels'
const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorfold-spine-durable-'))
const key='__spineDurableFixture'
const state:any={store:{settings:{defaultCwd:'',whatsapp:{sources:[{id:'synthetic-account',enabled:true,paired:true,phoneNumber:'15550000001'}]}},connectorEvents:[],projects:[],missions:[],projectMemoryCandidates:[]},writes:[],fail:false}
;(globalThis as any)[key]=state
try{
 const fixture=path.join(tmp,'fixture.mjs')
 await fs.writeFile(fixture,`const s=globalThis.${key};export const store=s.store;export const saveConnectorEvents=async(e)=>{if(s.fail)throw new Error('Synthetic event write failure');s.writes.push(structuredClone(e));};export const persistProjectMemoryCandidates=()=>{};export const auditEvent=async()=>{};`)
 const jiti=createJiti(__filename,{alias:{'../state':fixture,'../../store':fixture,'./persistence':fixture,'./audit':fixture},moduleCache:false,fsCache:false})
 const {createConnectorEventSpine}=await jiti.import<any>(path.join(__dirname,'connector-event-spine.ts'))
 const spine=createConnectorEventSpine({callbacks:{findApprovalRule:async()=>({decision:'allow'}),remotePublish:()=>{},emitOutboundWebhook:()=>{}}})
 const raw=personalMessageDraftEvent({connectorId:'whatsapp',target:'15550000000@lid',accountId:'synthetic-account',body:'Synthetic exact message'})
 const recorded=await spine.recordMessageEvent(raw)
 assert.equal(recorded.approvalStatus,'pending','old broad saved allow never silently approves a changed outbound body')
 assert.ok(recorded.reviewDigest)
 assert.equal(state.writes.length,1,'recorded approval is durable before the tool reports success')
 state.store.connectorEvents=[{...recorded,requiresApproval:false,approvalStatus:'approved',delivery:{transport:'whatsapp',status:'delivered'}}]
 await spine.rememberConnectorEvents([raw])
 assert.equal(state.store.connectorEvents.length,1,'re-ingest of the same host event ID never resurrects a pending approval')
 assert.equal(state.store.connectorEvents[0].delivery.status,'delivered')
 state.fail=true
 const another=personalMessageDraftEvent({connectorId:'whatsapp',target:'15550000000@lid',accountId:'synthetic-account',body:'Synthetic other message'})
 await assert.rejects(()=>spine.recordMessageEvent(another),/event write failure/,'caller cannot report a durable pending draft when the protected store rejects it')
 state.fail=false
 const recovered=await spine.recordMessageEvent(another)
 assert.equal(recovered.approvalStatus,'pending')
 assert.ok(state.writes.at(-1).some((event:any)=>event.id===another.id),'same host-event retry persists the retained exact draft after storage recovers')
 console.log('actual connector spine awaits durable host review, prevents broad auto-allow and consumed-event resurrection, and surfaces protected write failures')
}finally{delete(globalThis as any)[key];await fs.rm(tmp,{recursive:true,force:true})}
