import assert from 'node:assert/strict'
import { buildPersonalRoutine } from '../../../shared/personal-routines'
import { sanitizeAutomationCreateInput } from './automation-patch-policy'
import { assertRoutineSources, personalRoutinePrompt, routineSourceRevision } from './personal-routine-context'
import { connectorSourceBinding } from '../../agent/connector-source-binding'
import { defaultSettings } from '../../store'
import type { Automation, ConnectorEvent } from '../../agent/types'
const settings = defaultSettings()
settings.providers=[{id:'synthetic-local',name:'Synthetic',kind:'local',baseUrl:'http://127.0.0.1:1/v1',apiKey:'',models:['synthetic-model']}] as any
settings.agentPresets=[{id:'synthetic-bot',name:'Synthetic helper',role:'Review selected items',lane:'operations',providerId:'synthetic-local',model:'synthetic-model',instructions:'Be concise.',approvalScope:'readonly',executionTarget:'workspace',paused:false,createdAt:1,updatedAt:1}]
settings.documents.sources=[{id:'docs-1',enabled:true,kind:'dir',path:'/tmp/synthetic-docs-current'}] as any
const draft=buildPersonalRoutine({kind:'morning-brief',botId:'synthetic-bot',providerId:'synthetic-local',model:'synthetic-model',sourceScope:[{connectorId:'docs',sourceId:'docs-1'}],timeZone:'Europe/London'})
const safe=sanitizeAutomationCreateInput(draft)
assert.equal(safe.sourceScope?.[0].connectorId,'docs','recipe wire ID must pass the actual host sanitizer')
const automation={id:'synthetic-routine',everyMinutes:60,lastRunAt:null,nextRunAt:null,lastStatus:null,runs:[],createdAt:1,updatedAt:1,...safe} as Automation
assert.doesNotThrow(()=>assertRoutineSources(automation,settings),'docs source scope resolves settings.documents, retaining canonical wire ID')
assert.throws(()=>sanitizeAutomationCreateInput({botId:'synthetic-bot',mode:'agent'}),/Personal routines require/,'bot identity alone cannot decorate an unrestricted automation')
assert.throws(()=>sanitizeAutomationCreateInput({sourceScope:[],mode:'agent'}),/Personal routines require/,'source scope is enforced rather than decorative')
assert.throws(()=>assertRoutineSources({...automation,routineKind:undefined,mode:'agent'},settings),/Repair/,'persisted partial routine fields fail closed before global tools')
assert.equal(automation.enabled,false)
assert.equal(automation.deliverTo,'inbox')
assert.equal(automation.previewOnly,true)
const event=(id:string,connectorId:ConnectorEvent['connectorId'],sourceId:string,summary:string):ConnectorEvent=>({id,connectorId,pluginName:connectorId,action:'message_received',direction:'inbound',classification:'communication',priority:'normal',trustLevel:'suggest',requiresApproval:false,approvalId:null,title:'Synthetic item',summary,actor:'synthetic',threadKey:`${connectorId}:${sourceId}:item`,payloadKeys:[],sources:[{id:`ref-${id}`,connectorId,sourceId,title:'Synthetic source'}],createdAt:1})
const selected=event('allowed','docs','docs-1','SELECTED_SOURCE_CANARY')
selected.sources![0].path='/tmp/synthetic-docs-current/note.txt'
const other=event('other','docs','docs-2','UNSELECTED_SOURCE_CANARY')
const mixed={...selected,id:'mixed',summary:'MIXED_SOURCE_CANARY',sources:[...selected.sources!,...other.sources!]}
const prompt=personalRoutinePrompt(automation,settings,[selected,other,mixed])
assert.match(prompt,/SELECTED_SOURCE_CANARY/)
assert.doesNotMatch(prompt,/UNSELECTED_SOURCE_CANARY|MIXED_SOURCE_CANARY/,'unselected or mixed-source summaries never become model context')
const staleFile={...selected,id:'stale-file',summary:'STALE_FILE_ROOT_CANARY',sources:[{...selected.sources![0],path:'/tmp/synthetic-docs-prior/private.txt'}]}
assert.doesNotMatch(personalRoutinePrompt(automation,settings,[staleFile]),/STALE_FILE_ROOT_CANARY/,'reusing a source ID cannot authorize cached text from its old directory')
const sibling={...staleFile,summary:'SIBLING_ROOT_CANARY',sources:[{...selected.sources![0],path:'/tmp/synthetic-docs-current-evil/private.txt'}]}
assert.doesNotMatch(personalRoutinePrompt(automation,settings,[sibling]),/SIBLING_ROOT_CANARY/,'path-prefix siblings are outside the selected root')
for(const local of [
  {connectorId:'email',key:'email',kind:'eml_dir',path:'/tmp/synthetic-mail-current',valid:'/tmp/synthetic-mail-current/new.eml',old:'/tmp/synthetic-mail-prior/old.eml'},
  {connectorId:'calendar',key:'calendar',kind:'ics_file',path:'/tmp/synthetic-current.ics',valid:'/tmp/synthetic-current.ics',old:'/tmp/synthetic-prior.ics'},
  {connectorId:'browser',key:'browser',kind:'current_tab',url:'https://current.example.test/page',valid:'https://current.example.test/page',old:'https://prior.example.test/private'},
] as const){
  const source={id:'location-1',enabled:true,...local}
  ;(settings as any)[local.key].sources=[source]
  const routine={...automation,sourceScope:[{connectorId:local.connectorId,sourceId:'location-1'}]} as Automation
  const old=event('old-local',local.connectorId,'location-1','OLD_LOCATION_CANARY')
  old.sources![0].path=local.old
  assert.doesNotMatch(personalRoutinePrompt(routine,settings,[old]),/OLD_LOCATION_CANARY/,'old '+local.connectorId+' location is outside a reused source ID')
  const fresh={...old,summary:'CURRENT_LOCATION_CANARY',sources:[{...old.sources![0],path:local.valid}]}
  assert.match(personalRoutinePrompt(routine,settings,[fresh]),/CURRENT_LOCATION_CANARY/)
  const unbound={...fresh,summary:'UNBOUND_LOCAL_CANARY',sources:[{...old.sources![0],path:undefined}]}
  assert.doesNotMatch(personalRoutinePrompt(routine,settings,[unbound]),/UNBOUND_LOCAL_CANARY/)
}
const imapSource={id:'imap-1',enabled:true,kind:'imap',path:'',host:'mail.example.test',user:'owner@example.test',mailbox:'INBOX'}
settings.email.sources=[imapSource] as any
const imapRoutine={...automation,sourceScope:[{connectorId:'email',sourceId:'imap-1'}]} as Automation
const imap=event('imap-bound','email','imap-1','IMAP_CURRENT_CANARY')
imap.sources![0].path='imap://mail.example.test/INBOX/123'
imap.sources![0].sourceRevision=connectorSourceBinding(imapSource)
assert.match(personalRoutinePrompt(imapRoutine,settings,[imap]),/IMAP_CURRENT_CANARY/)
imapSource.user='different@example.test'
assert.doesNotMatch(personalRoutinePrompt(imapRoutine,settings,[imap]),/IMAP_CURRENT_CANARY/,'same host/mailbox with a different user cannot authorize prior account cache')
const historySource={id:'browser-1',enabled:true,kind:'history_export',path:'/tmp/synthetic-history-current.json'}
settings.browser.sources=[historySource] as any
const historyRoutine={...automation,sourceScope:[{connectorId:'browser',sourceId:'browser-1'}]} as Automation
const history=event('history-bound','browser','browser-1','HISTORY_CURRENT_CANARY')
history.sources![0].path='https://example.test/history-page'
history.sources![0].sourceRevision=connectorSourceBinding(historySource)
assert.match(personalRoutinePrompt(historyRoutine,settings,[history]),/HISTORY_CURRENT_CANARY/)
historySource.path='/tmp/synthetic-history-prior.json'
assert.doesNotMatch(personalRoutinePrompt(historyRoutine,settings,[history]),/HISTORY_CURRENT_CANARY/,'history source file identity remains bound independently of each visited URL')
settings.whatsapp.sources=[{id:'wa-1',enabled:true,paired:true,phoneNumber:'15550000001',allowedChatJids:['15550000002@lid'],allowSelfChat:true}] as any
const wa={...automation,sourceScope:[{connectorId:'whatsapp',sourceId:'wa-1'}]} as Automation
const inbound=(jid:string,text:string,overrides:any={})=>({...event(text,'whatsapp','wa-1',text),threadKey:`whatsapp:wa-1:${jid}`,payload:{jid,authenticatedAccountJid:'15550000001'+'@s.whatsapp.net',...overrides}})
const current=inbound('15550000002@lid','CURRENT_CONTACT_CANARY')
const revoked=inbound('15550000003@lid','REVOKED_CONTACT_CANARY')
const self=inbound('15550000001@lid','SELF_CHAT_CANARY',{selfChat:true})
const oldPhone=inbound('15550000002@lid','OLD_PHONE_CANARY',{authenticatedAccountJid:'15550000009'+'@s.whatsapp.net'})
const legacy=inbound('15550000002@lid','UNBOUND_OLD_CANARY',{authenticatedAccountJid:undefined})
const scoped=personalRoutinePrompt(wa,settings,[current,revoked,self,oldPhone,legacy])
assert.match(scoped,/CURRENT_CONTACT_CANARY|SELF_CHAT_CANARY/)
assert.doesNotMatch(scoped,/REVOKED_CONTACT_CANARY|OLD_PHONE_CANARY|UNBOUND_OLD_CANARY/)
const prior=routineSourceRevision(wa,settings)
settings.whatsapp.sources[0].allowedChatJids=[]
assert.notEqual(routineSourceRevision(wa,settings),prior,'scope edits invalidate an in-flight captured source revision')
assert.doesNotMatch(personalRoutinePrompt(wa,settings,[current]),/CURRENT_CONTACT_CANARY/)
// Re-apply every current cached conversation grant; a source ID alone is not consent
// to contacts/channels removed after an earlier sync.
const grantCases:any[]=[
  {connectorId:'imessage',sourceId:'imessage',settings:{enabled:true,allowedChatIds:'allowed@example.test'},threadKey:'imessage:blocked@example.test',payload:{handle:'blocked@example.test',chat:'blocked@example.test'}},
  {connectorId:'slack',sourceId:'slack-1',source:{id:'slack-1',enabled:true,kind:'api_token',channels:'allowed-channel'},threadKey:'slack:slack-1:blocked-channel:1',payload:{channel:'blocked-channel'}},
  {connectorId:'discord',sourceId:'discord-1',source:{id:'discord-1',enabled:true,allowedChannels:'allowed-channel'},threadKey:'discord:discord-1:blocked-channel',payload:{channel:'blocked-channel',thread:null}},
  {connectorId:'mattermost',sourceId:'mm-1',source:{id:'mm-1',enabled:true,allowedChannels:'allowed-channel'},threadKey:'mattermost:mm-1:blocked-channel',payload:{channel:'blocked-channel',thread:null}},
  {connectorId:'matrix',sourceId:'matrix-1',source:{id:'matrix-1',enabled:true,allowedRooms:'!allowed:example.test'},threadKey:'matrix:matrix-1:!blocked:example.test',payload:{room:'!blocked:example.test',thread:null}},
  {connectorId:'googlechat',sourceId:'google-1',source:{id:'google-1',enabled:true,allowedChannels:'spaces/allowed'},threadKey:'googlechat:google-1:spaces/blocked',payload:{space:'spaces/blocked',thread:null}},
]
for(const item of grantCases){
  ;(settings as any)[item.connectorId]=item.settings??{sources:[item.source]}
  const row={...event('blocked-'+item.connectorId,item.connectorId,item.sourceId,'BLOCKED_GRANT_CANARY_'+item.connectorId),threadKey:item.threadKey,payload:item.payload}
  const routine={...automation,sourceScope:[{connectorId:item.connectorId,sourceId:item.sourceId}]} as Automation
  assert.doesNotMatch(personalRoutinePrompt(routine,settings,[row]),/BLOCKED_GRANT_CANARY/,'cached '+item.connectorId+' messages from revoked conversations stay out of model context')
  const allowedRow=JSON.parse(JSON.stringify(row).replaceAll('blocked','allowed').replaceAll('BLOCKED_GRANT','ALLOWED_GRANT'))
  assert.match(personalRoutinePrompt(routine,settings,[allowedRow]),/ALLOWED_GRANT_CANARY/,'current '+item.connectorId+' grants preserve eligible cached context')
  const malformed={...allowedRow,threadKey:row.threadKey,summary:'MALFORMED_BINDING_CANARY'}
  if(item.connectorId!=='imessage')assert.doesNotMatch(personalRoutinePrompt(routine,settings,[malformed]),/MALFORMED_BINDING_CANARY/,'cached metadata cannot change '+item.connectorId+' conversation identity')
  const grantOwner=item.source??(settings as any)[item.connectorId]
  const grantField=item.connectorId==='imessage'?'allowedChatIds':item.connectorId==='matrix'?'allowedRooms':item.connectorId==='slack'?'channels':'allowedChannels'
  const goodGrant=grantOwner[grantField]
  for(const malformedGrant of [null,[],{},42]) { grantOwner[grantField]=malformedGrant;assert.doesNotMatch(personalRoutinePrompt(routine,settings,[allowedRow]),/ALLOWED_GRANT_CANARY/,'malformed '+item.connectorId+' grants cannot mean unrestricted') }
  grantOwner[grantField]=goodGrant
  const before=routineSourceRevision(routine,settings)
  if(item.source){ const field=item.connectorId==='matrix'?'allowedRooms':item.connectorId==='slack'?'channels':'allowedChannels';item.source[field]='different-grant' }
  else (settings as any)[item.connectorId].allowedChatIds='other@example.test'
  assert.notEqual(routineSourceRevision(routine,settings),before,'in-flight '+item.connectorId+' consent changes invalidate captured context')
}
const bounded=personalRoutinePrompt(automation,settings,Array.from({length:80},(_,i)=>({...selected,id:'bounded-'+i,summary:'Large selected text '.repeat(200)})),{maxCharacters:3_000})
assert.ok(bounded.length<=3_000)
assert.match(bounded,/older items omitted|Excerpts may be shortened/)
assert.throws(()=>personalRoutinePrompt(automation,settings,[selected],{maxCharacters:100}),/instructions exceed/)
settings.agentPresets[0].paused=true
assert.throws(()=>personalRoutinePrompt(wa,settings,[self]),/Resume this bot/)
settings.agentPresets[0].paused=false
assert.throws(()=>personalRoutinePrompt({...wa,mode:'agent'},settings,[]),/read-only|Repair/,'routine source metadata never permits the global agent tool context')
assert.throws(()=>personalRoutinePrompt({...wa,model:'other-model'},settings,[]),/pinned model/)
console.log('actual recipe-to-host docs mapping, inbox preview defaults, exact source/model/bot boundaries and cached WhatsApp current-consent filtering passed')
