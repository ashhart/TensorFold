import assert from 'node:assert/strict'
import { ClusterWorkspaceStore, emptyClusterState } from './store'
const token='synthetic_management_credential_'.repeat(3)
let data=emptyClusterState(),writes=0,fail=false,current=true,release!:()=>void,hold=false
const observations:any[]=[]
const store=new ClusterWorkspaceStore({profile:'/synthetic/profile',assertCurrent(){if(!current)throw new Error('Synthetic profile changed')},read:async()=>structuredClone(data),write:async(next,guard)=>{if(hold)await new Promise<void>(done=>{release=done});guard();if(fail)throw new Error('Synthetic durable disk failure');data=structuredClone(next);writes++},observe:value=>observations.push(structuredClone(value))})
const host:any={id:'host-a',revision:1,label:'Synthetic',role:'headless',connection:{kind:'endpoint',url:'https://synthetic.example.invalid'},nodeId:null,hasCredential:true,snapshot:null,state:'unverified',lastError:null,createdAt:1,updatedAt:1,token,sshStamp:null}
fail=true;await assert.rejects(()=>store.transaction(state=>state.hosts.push(host)),/disk failure/);assert.equal(data.revision,0);assert.equal(data.hosts.length,0);assert.ok(!observations.some(state=>state.hosts.length));fail=false
await store.transaction(state=>state.hosts.push(host));const publicState=await store.publicSnapshot();assert.ok(!JSON.stringify(publicState).includes(token));assert.equal('sshStamp'in publicState.hosts[0],false);assert.equal('token'in publicState.hosts[0],false)
hold=true;const pending=store.transaction(state=>{state.hosts[0].label='Cannot cross profile'});for(let i=0;i<10&&!release;i++)await Promise.resolve();current=false;release();await assert.rejects(()=>pending,/profile changed/);assert.equal(data.hosts[0].label,'Synthetic');current=true;hold=false
await Promise.all([store.transaction(state=>{state.hosts[0].revision++}),store.transaction(state=>{state.hosts[0].revision++})]);assert.equal(data.hosts[0].revision,3,'Serialized transactions cannot drop a concurrent revision')
const before=structuredClone(data)
for(const malformed of [
 {evidence:[{planId:'missing-plan',members:[{hostId:'host-a',digest:'not-a-digest'}]}]},
 {previews:[{public:{id:'bad-preview',digest:'not-a-digest',expiresAt:'not-a-time'},internal:{},appliedCommand:null,state:'preview'}]},
]){
 data={...structuredClone(before),...malformed} as any
 await assert.rejects(()=>store.get(),/invalid|preview|evidence|saved|digest|number|cluster/i,'Corrupt nested durable evidence/setup data must be refused rather than accepted and overwritten by a later transaction')
}
console.log('Protected cluster store with fake persistence: no publication on write failure, public secret omission, profile fences, serialized revisions and strict nested evidence/setup validation passed')
