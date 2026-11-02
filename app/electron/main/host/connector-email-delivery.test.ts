import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { emailDraftEvent } from '../../agent/email-connector'
const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorfold-email-delivery-'))
const key='__emailDeliveryFixture'
const state:any={store:{settings:{email:{sources:[{id:'synthetic-smtp',kind:'imap',enabled:true,host:'mail.example.test',user:'sender@example.test',fromAddress:'sender@example.test',password:'synthetic-password',port:587}]}},connectorEvents:[]},smtp:[],outbox:[],graph:[],draft:emailDraftEvent}
;(globalThis as any)[key]=state
try{
  const fixture=path.join(tmp,'fixture.mjs')
  await fs.writeFile(fixture,`const s=globalThis.${key};export const store=s.store;export const saveConnectorEvents=async()=>{};export const vontraPath=(...p)=>p.join('/');export const auditEvent=async()=>{};export const microsoftMailRuntime=()=>({send:async(account,event)=>{s.graph.push({account,event});return {transport:'microsoft_graph',status:'queued',deliveredAt:null,error:null}}});export const emailDraftEvent=(...p)=>s.draft(...p);export const sendEmailOverSmtp=async(e,a)=>{s.smtp.push({event:e,account:a});return {transport:'smtp',messageId:'synthetic-smtp-id',deliveredAt:123}};export const writeEmailOutboxMessage=async(e)=>{s.outbox.push(e);return {transport:'local_outbox',path:'synthetic-outbox.eml',deliveredAt:123}};`)
  const jiti=createJiti(__filename,{alias:{'../state':fixture,'../../store':fixture,'./audit':fixture,'../../agent/smtp-send':fixture,'../../agent/email-connector':fixture,'../../agent/email-provider-runtime':fixture},moduleCache:false,fsCache:false})
  const {createEmailTransport}=await jiti.import<any>(path.join(__dirname,'connector-email-transport.ts'))
  const {createReviewedConnectorDelivery,sealConnectorOutboundEvent}=await jiti.import<any>(path.join(__dirname,'connector-outbound-delivery.ts'))
  const email=createEmailTransport({spine:{rememberConnectorEvents:async()=>{},recordEventDelivery:async()=>{}}})
  const delivery=createReviewedConnectorDelivery({adapters:[],sendEmail:email.deliverApprovedEmailEvent,persist:async()=>{}})
  const make=(accountId?:string)=>{
    const event=sealConnectorOutboundEvent(emailDraftEvent({to:'recipient@example.test',subject:'Exact subject',body:'  Exact reviewed email body.  ',mode:'send',accountId}))
    state.store.connectorEvents=[event];return event
  }
  const smtp=make('synthetic-smtp')
  const sent=await delivery.approve(smtp,smtp.reviewDigest,'synthetic-owner')
  assert.equal(sent.delivery.status,'delivered')
  assert.equal(sent.delivery.transport,'smtp')
  assert.equal(state.smtp.length,1)
  assert.equal(state.smtp[0].account.id,'synthetic-smtp')
  assert.equal(state.smtp[0].event.payload.to,'recipient@example.test')
  assert.equal(state.smtp[0].event.payload.body,'  Exact reviewed email body.  ')
  assert.equal(state.smtp[0].event.payload.subject,'Exact subject')
  const local=make()
  const queued=await delivery.approve(local,local.reviewDigest,'synthetic-owner')
  assert.equal(queued.delivery.status,'queued','a local .eml file is queued, never called recipient delivery')
  assert.equal(queued.delivery.deliveredAt,null)
  assert.equal(state.smtp.length,1)
  assert.equal(state.outbox.length,1)
  const changed=make('synthetic-smtp')
  state.store.settings.email.sources[0].user='other@example.test'
  await assert.rejects(()=>delivery.approve(changed,changed.reviewDigest,'synthetic-owner'),/changed|not fully reviewed/)
  assert.equal(state.smtp.length,1,'SMTP sender changes cannot inherit the old exact review')
  const invalid=make('missing-account')
  assert.match(invalid.outboundError,/enabled email account/)
  await assert.rejects(()=>email.deliverApprovedEmailEvent({...local,approvalStatus:'approved',requiresApproval:false}),/durable exact-content approval/,'direct email transport rejects approval without a durable sending marker')
  state.store.settings.email.sources.push({id:'synthetic-graph',kind:'microsoft_graph',enabled:true,user:'account@example.com',microsoftRevision:1})
  const graph=make('synthetic-graph')
  const accepted=await delivery.approve(graph,graph.reviewDigest,'synthetic-owner')
  assert.equal(accepted.delivery.transport,'microsoft_graph')
  assert.equal(accepted.delivery.status,'queued')
  assert.equal(state.graph.length,1,'actual reviewed email dispatcher selects Microsoft transport, never SMTP or a local export')
  assert.equal(state.graph[0].event.outbound.body,'  Exact reviewed email body.  ')
  const graphChanged=make('synthetic-graph')
  state.store.settings.email.sources.at(-1).microsoftRevision=2
  await assert.rejects(()=>delivery.approve(graphChanged,graphChanged.reviewDigest,'synthetic-owner'),/changed|not fully reviewed/)
  assert.equal(state.graph.length,1,'Microsoft reauthorization cannot inherit a stale send approval')
  console.log('actual approved email dispatcher preserves selected SMTP identity/exact content, truthfully queues local outbox and rejects stale or missing durable reviews')
}finally{delete(globalThis as any)[key];await fs.rm(tmp,{recursive:true,force:true})}
