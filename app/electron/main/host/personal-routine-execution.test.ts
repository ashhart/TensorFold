import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { defaultSettings } from '../../store'
import { buildPersonalRoutine } from '../../../shared/personal-routines'
const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorfold-routine-execution-'))
const key='__routineExecutionFixture'
const settings=defaultSettings()
settings.providers=[{id:'synthetic-local',name:'Synthetic',kind:'local',baseUrl:'http://127.0.0.1:1/v1',apiKey:'',models:['synthetic-model']}] as any
settings.agentPresets=[{id:'synthetic-bot',name:'Synthetic bot',role:'Make a useful brief',lane:'operations',providerId:'synthetic-local',model:'synthetic-model',executionTarget:'workspace',approvalScope:'readonly',toolScope:{mode:'selected',names:[]},paused:false,createdAt:1,updatedAt:1}]
settings.slack.sources=[{id:'slack-1',kind:'api_token',enabled:true,channels:'C_ALLOWED'}] as any
const row=(channel:string,text:string)=>({id:channel,connectorId:'slack',pluginName:'Slack',action:'message_received',direction:'inbound',classification:'communication',priority:'normal',trustLevel:'notify',requiresApproval:false,approvalId:null,title:'Synthetic',summary:text,actor:'synthetic',threadKey:`slack:slack-1:${channel}:1`,payloadKeys:['channel'],payload:{channel},createdAt:1})
const state:any={store:{settings,connectorEvents:[row('C_ALLOWED','SCOPED_ALLOWED_CANARY'),row('C_OTHER','OUTSIDE_SCOPE_CANARY')],automations:[],jobs:[],missions:[],rooms:[]},requests:[],events:[],routes:[],beforeFresh:null,beforeYield:null,globalTools:0}
;(globalThis as any)[key]=state
try{
  const fixture=path.join(tmp,'fixture.mjs')
  await fs.writeFile(fixture,`const s=globalThis.${key};export const store=s.store;export const persistAutomations=()=>{};export const auditEvent=async()=>{};export const requireSecurityRole=()=>{};export const handle=()=>{};export const engineService=()=>null;export const runAgent=async()=>{throw new Error('Routine reached global agent tools')};export async function* streamChat(p){p.assertRequestAllowed?.();s.requests.push(p);await s.beforeYield?.();yield {type:'content',text:'Synthetic final brief'};}export const runtimeProviderWithFreshModels=async(p)=>{await s.beforeFresh?.();return p};export const selectedProviderAccountId=(p,a)=>a??null;export const modelRouter={select:(p)=>{s.routes.push(p);return {selected:{},selectedAccountId:p.requestedAccountId,requestedAccountId:p.requestedAccountId,audit:{action:'selected'},candidates:[]}},routeAfterError:()=>({signal:null,route:{selected:null}})};export const modelRouteAutoFailoverEnabled=()=>true;export const modelRouteAccountLabel=()=>'';export const modelRouteResetPolicy=()=>'';export const rememberModelRouteHistory=()=>{};export const routeHandoffPreview=(x)=>x;export const runtimeProvider=(p)=>p;`)
  const jiti=createJiti(__filename,{alias:{'./persistence':fixture,'./audit':fixture,'./ipc':fixture,'../state':fixture,'./engine-runtime':fixture,'./provider-runtime':fixture,'../../agent/loop':fixture,'../../agent/openai':fixture},moduleCache:false,fsCache:false})
  const {createAutomationRuntime}=await jiti.import<any>(path.join(__dirname,'automations.ts'))
  const runtime=createAutomationRuntime({threadJobHeartbeatMs:30_000,claimJob:()=>({id:'synthetic-job'}),patchJob:(job:any)=>job,heartbeatJob:(job:any)=>job,finishJob:()=>{},emitAutomation:()=>{},effectiveContextLimit:()=>8192,fullToolContext:()=>{state.globalTools++;throw new Error('Routine requested global context')},applyEventToThread:()=>{},timelineErrorSince:()=>null,timelineError:()=>new Error('synthetic'),isHighRiskTool:()=>false,rememberConnectorEvents:async(events:any[])=>state.events.push(...events)})
  const make=()=>runtime.mkAutomation(buildPersonalRoutine({kind:'morning-brief',botId:'synthetic-bot',providerId:'synthetic-local',model:'synthetic-model',sourceScope:[{connectorId:'slack',sourceId:'slack-1'}],timeZone:'Europe/London'}))
  const routine=make()
  await runtime.runAutomation(routine)
  assert.equal(routine.runs[0].status,'ok')
  assert.equal(state.requests.length,1)
  const request=state.requests[0]
  assert.deepEqual(request.tools,[],'actual runtime offers no tools to a personal routine')
  assert.equal(request.cwd,undefined,'routine does not inherit global working directory')
  assert.match(request.messages[1].content,/SCOPED_ALLOWED_CANARY/)
  assert.doesNotMatch(request.messages[1].content,/OUTSIDE_SCOPE_CANARY/)
  assert.equal(state.globalTools,0)
  assert.equal(state.routes[0].allowFailover,false,'pinned routine account cannot silently fail over')
  assert.equal(state.events.length,0,'default preview produces only local output')

  state.beforeFresh=()=>{settings.slack.sources[0].channels='C_OTHER'}
  const revoked=make()
  await runtime.runAutomation(revoked)
  assert.equal(revoked.runs[0].status,'error')
  assert.match(revoked.runs[0].error,/source or contact scope changed/)
  assert.equal(state.requests.length,1,'source-consent change during provider await blocks before model transport')
  settings.slack.sources[0].channels='C_ALLOWED'
  state.beforeFresh=()=>{settings.agentPresets[0].paused=true}
  const paused=make()
  await runtime.runAutomation(paused)
  assert.equal(paused.runs[0].status,'error')
  assert.equal(state.requests.length,1,'paused bot cannot reach model transport after an await')
  settings.agentPresets[0].paused=false;state.beforeFresh=null

  const external=make()
  external.previewOnly=false;external.deliverTo='telegram';external.deliveryTarget={connectorId:'telegram',accountId:'bot-control',recipient:'12345'}
  state.beforeYield=()=>{external.deliveryTarget.recipient='99999'}
  await runtime.runAutomation(external)
  assert.equal(external.runs[0].status,'ok')
  assert.equal(state.events[0].payload.target,'12345','an in-flight run keeps the captured explicit destination')
  assert.equal(state.events[0].payload.body,'Synthetic final brief','review uses the exact complete generated output')
  assert.equal(external.runs[0].deliveryStatus,'pending_approval')
  assert.equal(state.events[0].approvalStatus,'pending','generation cannot bypass the owner-reviewed delivery pipeline')
  console.log('actual routine runtime uses no tools/global context, captures model/destination, filters source grants and blocks consent/bot changes across provider awaits')
}finally{delete(globalThis as any)[key];await fs.rm(tmp,{recursive:true,force:true})}
