import assert from 'node:assert/strict'
import { ClusterWorkspaceStore, emptyClusterState } from './store'
import { ClusterManager } from './manager'
import { ClusterRemoteError } from './client'
import type { ClusterNodeSnapshot } from '../../../../shared/cluster-workspace'
const token='synthetic_management_credential_'.repeat(3),inference='synthetic_inference_credential_'.repeat(3),revision='a'.repeat(40)
const profile:any={id:'single',title:'Synthetic single profile',mode:'single',platform:'darwin',arch:'arm64',ranks:1,transports:['tcp'],modelTypes:['qwen3_5'],poolsWeights:false,requiresFabric:false,engineVersion:'1.0.12'}
const baseNode:ClusterNodeSnapshot={protocol:1,nodeId:'node-a',version:'1',platform:'darwin',arch:'arm64',observedAt:Date.now(),totalMemoryBytes:10000,freeMemoryBytes:8000,engine:{available:true,version:'1.0.12',profiles:[profile],reason:null},models:[{repo:'Example/Model',revision,modelType:'qwen3_5',weightsBytes:1024,memoryFloorBytes:2048}],links:[],active:null}
function fixture(){
 let data=emptyClusterState(),writes=0,failWrite=false,current=true,leases=0,effects=0,uncertain=false,wrongReceipt=false
 let node=structuredClone(baseNode),effectResolve!:()=>void,holdEffect=false,finalSnapshotFail=false,readCount=0
 data.hosts=[{id:'host-a',revision:1,label:'Synthetic A',role:'headless',connection:{kind:'endpoint',url:'https://synthetic.example.invalid'},token,sshStamp:null,nodeId:'node-a',hasCredential:true,snapshot:structuredClone(node),state:'ready',lastError:null,createdAt:1,updatedAt:1}]
 const remote=new Map<string,any>(),promoted:any[]=[],events:string[]=[]
 const store=new ClusterWorkspaceStore({profile:'/synthetic/profile',assertCurrent(){if(!current)throw new Error('Synthetic profile changed')},read:async()=>{readCount++;if(finalSnapshotFail&&effects===0&&data.jobs.length>0&&readCount>2){finalSnapshotFail=false;throw new Error('Synthetic final snapshot read failed')}return structuredClone(data)},write:async(next,guard)=>{guard();if(failWrite)throw new Error('Synthetic durable write failed');data=structuredClone(next);writes++;events.push('commit')}})
 const job=(kind:string,input:any)=>{const row={id:'remote-'+kind,commandId:wrongReceipt?'other-command':input.commandId,kind,phase:'completed',step:'finished',message:'Synthetic completed receipt',createdAt:1,updatedAt:2};remote.set(input.commandId,row);return row}
 const effect=async(kind:string,input:any)=>{
  assert.ok(data.remote.some(row=>row.commandId===input.commandId),'Remote effect is admitted only after its protected command receipt exists')
  events.push('effect');effects++
  if(holdEffect)await new Promise<void>(resolve=>{effectResolve=resolve})
  if(uncertain){job(kind,input);throw new ClusterRemoteError('Synthetic lost acknowledgement',true)}
  if(kind==='start')node.active={id:input.deploymentId,commandId:input.commandId,state:'ready',model:{repo:'Example/Model',revision},rank:0,profileId:'single',startedAt:1,error:null,endpointReady:false,canaryPassed:false}
  if(kind==='canary'){assert.ok(node.active);node.active!.canaryPassed=true;node.active!.endpointReady=true}
  if(kind==='stop'&&node.active)node.active={...node.active,state:'stopped'}
  return job(kind,input)
 }
 const client:any={baseUrl:'https://synthetic.example.invalid',status:async()=>structuredClone(node),admission:async()=>({allowed:true,nodeId:'node-a',requiredBytes:2048,budgetBytes:8000,reason:'Synthetic fits',cached:true,profile}),ensure:(input:any)=>effect('ensure-model',input),start:(input:any)=>effect('start',input),stop:(input:any)=>effect('stop',input),canary:(input:any)=>effect('canary',input),credentials:async()=>({apiKey:inference,model:'Example/Model'}),job:async(id:string)=>[...remote.values()].find(row=>row.id===id),jobs:async()=>[...remote.values()]}
 const manager=new ClusterManager({store,assertCurrent(){if(!current)throw new Error('Synthetic profile changed')},client:async()=>client,sshStamp:async()=> 'b'.repeat(64),invalidate:async()=>{},reserve(){leases++;let released=false;return{release(){if(!released){released=true;leases--}}}},useEndpoint:async value=>{promoted.push(value);return{providerId:'synthetic-managed-provider'}},delay:async()=>{}})
 const input:any={name:'Synthetic plan',model:{repo:'Example/Model',revision},profileId:'single',generation:{contextTokens:8192,maxOutputTokens:1024,temperature:0.1},members:[{hostId:'host-a',hostRevision:1,nodeId:'node-a',rank:0}],transport:'tcp'}
 return{manager,store,input,client,events,promoted,data:()=>data,node:()=>node,effects:()=>effects,leases:()=>leases,writes:()=>writes,setFail(value:boolean){failWrite=value},setUncertain(){uncertain=true},setWrongReceipt(){wrongReceipt=true},hold(){holdEffect=true},release(){effectResolve()},mutateNode(patch:any){Object.assign(node,patch)},failFinalSnapshot(){readCount=0;finalSnapshotFail=true},loseScope(){current=false}}
}
let h=fixture(),plan=await h.manager.plan(h.input)
assert.equal(plan.blockers.length,0);assert.equal(plan.stageAllowed,true)
let before=h.effects();h.setFail(true);await assert.rejects(()=>h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-command-ensure'}),/write failed/);assert.equal(h.effects(),before);assert.equal(h.leases(),0);h.setFail(false)
await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-command-ensure'});await h.manager.settle();assert.equal(h.data().jobs[0].phase,'completed');assert.equal(h.effects(),1);assert.equal(h.leases(),0)
await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-command-ensure'});await h.manager.settle();assert.equal(h.effects(),1,'Repeated durable command cannot repeat an effect')
await assert.rejects(()=>h.manager.begin('activate',{planId:plan.id,digest:plan.digest,commandId:'synthetic-command-ensure'}),/different operation/)
h=fixture();plan=await h.manager.plan(h.input);h.mutateNode({models:[{...baseNode.models[0],weightsBytes:4096}]});await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-stale-command'});await h.manager.settle();assert.equal(h.effects(),0);assert.equal(h.data().jobs[0].phase,'failed','Changed checkpoint/engine evidence refuses before any effect')
h=fixture();plan=await h.manager.plan(h.input);h.setUncertain();await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-lost-command'});await h.manager.settle();assert.equal(h.data().jobs[0].phase,'needs-review');assert.equal(h.data().remote[0].nodeJobId,null)
await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-lost-command'});assert.equal(h.effects(),1)
await h.manager.reconcile({jobId:h.data().jobs[0].id});assert.equal(h.data().jobs[0].phase,'completed');assert.equal(h.effects(),1,'Reconciliation finds the durable remote receipt without replay')
h=fixture();plan=await h.manager.plan(h.input);h.hold();h.manager.snapshot=async()=>{throw new Error('Synthetic response snapshot failed after launch')}
await assert.rejects(()=>h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-response-read-fail'}),/snapshot failed/)
for(let i=0;i<50&&h.effects()===0;i++)await Promise.resolve()
assert.equal(h.effects(),1);assert.equal(h.leases(),1,'The launched operation retains its physical lease even when the initial API snapshot response fails')
h.release();await h.manager.settle();assert.equal(h.leases(),0);assert.equal(h.data().jobs[0].phase,'completed')
h=fixture();plan=await h.manager.plan(h.input);h.setWrongReceipt();await h.manager.begin('ensure-model',{planId:plan.id,digest:plan.digest,commandId:'synthetic-wrong-receipt'});await h.manager.settle();assert.equal(h.data().jobs[0].phase,'needs-review')
h=fixture();plan=await h.manager.plan(h.input);await h.manager.begin('activate',{planId:plan.id,digest:plan.digest,commandId:'synthetic-activate-command'});await h.manager.settle();assert.equal(h.data().jobs[0].phase,'completed','A matching ready rank plus canary must commit an endpoint')
const deployment=h.data().deployments[0];assert.equal(deployment.state,'ready');assert.equal(new URL(deployment.endpoint!.baseUrl).pathname,'/v1');assert.equal(deployment.endpoint!.location,'remote')
const creds=await h.manager.credentials(deployment.id);assert.equal(new URL(creds.baseUrl).pathname,'/v1');await h.manager.use(deployment.id);assert.equal(h.promoted.length,1)
const exactActive=structuredClone(h.node().active!)
for(const patch of [{id:'other-deployment'},{model:{repo:'Other/Model',revision}},{model:{repo:'Example/Model',revision:'c'.repeat(40)}},{rank:1},{profileId:'other-profile'},{endpointReady:false},{canaryPassed:false},{state:'starting'}]){
 h.mutateNode({active:{...exactActive,...patch}});await assert.rejects(()=>h.manager.credentials(deployment.id),/deployment changed/);assert.equal(h.promoted.length,1,'No changed deployment is silently re-promoted')
}
h.mutateNode({active:exactActive})
await h.store.recover();assert.equal(h.data().deployments[0].state,'needs-review');assert.equal(h.data().deployments[0].endpoint!.ready,false);await assert.rejects(()=>h.manager.credentials(deployment.id),/Reconcile/)
h=fixture()
const sshInput:any={expectedRevision:0,label:'Synthetic SSH',role:'headless',connection:{kind:'ssh',sshHostId:'ssh-a',port:8766},token}
const beforeEnrollment=h.writes();await assert.rejects(()=>h.manager.save(sshInput,'c'.repeat(64)),/enrollment target changed/);assert.equal(h.writes(),beforeEnrollment,'A bootstrap identity mismatch cannot persist its management credential')
await h.manager.save(sshInput);let saved=h.data().hosts.find(row=>row.connection.kind==='ssh')!;assert.equal(saved.token,token)
h.manager.options.sshStamp=async()=> 'c'.repeat(64)
await h.manager.save({id:saved.id,expectedRevision:saved.revision,label:saved.label,role:saved.role,connection:saved.connection});saved=h.data().hosts.find(row=>row.id===saved.id)!
assert.equal(saved.token,null);assert.equal(saved.sshStamp,null);assert.equal(saved.hasCredential,false,'Reconfigured SSH aliases never inherit another target’s management credential')
console.log('Cluster manager/store/operations with fake adapters: durable before effect, failed writes, payload replay, stale evidence, uncertain receipts, reconciliation, exact deployment identity, SSH credential rebinding and restart gates passed')
