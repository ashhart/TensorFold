import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-automation-delivery-'))
const previous = process.env.VONTRA_HOME
process.env.VONTRA_HOME = tmp
try {
  const { defaultSettings } = await import('../../store')
  const { store } = await import('../state')
  const { createAutomationRuntime } = await import('./automations')
  store.settings = defaultSettings()
  const events: any[] = []
  const runtime = createAutomationRuntime({
    threadJobHeartbeatMs:30_000, claimJob:()=>({id:'synthetic-job'}) as any,
    patchJob:(job:any)=>job, heartbeatJob:(job:any)=>job, finishJob:()=>{}, emitAutomation:()=>{},
    effectiveContextLimit:()=>8192, fullToolContext:()=>{throw new Error('No global tool access for a digest fixture')},
    applyEventToThread:()=>{}, timelineErrorSince:()=>null, timelineError:()=>new Error('synthetic'), isHighRiskTool:()=>false,
    rememberConnectorEvents:async incoming=>{events.push(...incoming)},
  })
  const automation = runtime.mkAutomation({name:'Synthetic phone brief',mode:'digest',deliverTo:'whatsapp' as any,deliveryTarget:{connectorId:'whatsapp',accountId:'synthetic-wa',recipient:'15550000000@lid'},previewOnly:false} as any)
  await runtime.runAutomation(automation)
  assert.equal(automation.runs[0].status,'ok')
  assert.equal(events.length,1)
  assert.equal(events[0].payload?.body,automation.runs[0].output,'result review must contain full exact generated body, not only summary/key names')
  assert.equal(events[0].payload?.accountId,'synthetic-wa','destination binds an explicit selected account')
  assert.equal(events[0].payload?.target,'15550000000@lid','destination binds an explicit reviewed recipient')
  assert.equal(events[0].action,'message_send_requested','automation uses the existing addressed send approval pipeline')
  assert.equal(events[0].approvalStatus,'pending','finished generation never implies delivery')
  assert.equal(automation.runs[0].deliveryStatus,'pending_approval')
  const preview = runtime.mkAutomation({...automation,id:undefined,previewOnly:true} as any)
  await runtime.runAutomation(preview)
  assert.equal(events.length,1,'preview mode retains output locally and never queues an external send')
  assert.equal(preview.runs[0].deliveryStatus,'inbox')
  console.log('actual automation output-to-reviewed-destination contract passed')
} finally {
  const { writeChains } = await import('../../store')
  await Promise.allSettled([...writeChains.values()])
  if (previous===undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME=previous
  await fs.rm(tmp,{recursive:true,force:true})
}
