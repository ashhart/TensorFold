import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import vm from 'node:vm'
import { createClusterSshSetup, type ClusterBootstrapRecord } from './ssh-setup'
import { SSH_INSTALL_PROGRAM, SSH_PREREQUISITE_PROGRAM, SSH_CREDENTIAL_PROGRAM } from './ssh-install-script'
for (const program of [SSH_INSTALL_PROGRAM,SSH_PREREQUISITE_PROGRAM,SSH_CREDENTIAL_PROGRAM]) new vm.Script(program)
assert.doesNotMatch(SSH_INSTALL_PROGRAM,/\bsudo\b|authorized_keys|ssh-keygen|\/etc\/systemd|enable-linger|kextload/)
const token='synthetic_management_credential_'.repeat(3), bytes=Buffer.from('module.exports = { synthetic: true };'), bundle={bytes,sha256:createHash('sha256').update(bytes).digest('hex'),version:'1' as const,entry:'cluster-node.js' as const}
const baseline={platform:'linux',arch:'arm64',home:'/home/synthetic',uid:1001,nodePath:'/synthetic/bin/node',nodeVersion:'22.23.2',enginePath:'/synthetic/bin/tensorfold-native',engineVersion:'TensorFold 1.0.12',supervisor:'systemd-user',serviceReady:true,lingering:true,homeOwned:true,serviceExists:false,unsafeExisting:false,existing:null}
function fixture() {
 let missingNode=false
 let profile='/synthetic/profile',host:any={id:'synthetic-host',hostName:'synthetic.local',port:22,user:'synthetic',identityFiles:[],origin:'manual',createdAt:1,updatedAt:1},report={...baseline},persistFailure=false,installFailure=false,verificationFailure=false,invalidProof=false
 const records=new Map<string,ClusterBootstrapRecord>(),states:any[]=[],calls:any[]=[],events:string[]=[]
 const session={host:structuredClone(host),onClosed(){return()=>{}},async forwardLoopback(){throw new Error('No real forwarding in setup tests')},async exec(command:string,options:any={}){
  calls.push({command,options})
  if(missingNode)return{code:42,stdout:'TENSORFOLD_NODE_MISSING\nLinux\n',stderr:''}
  if(options.input!==undefined){events.push('remote-mutation');assert.equal(states.at(-1)?.state,'applying');const payload=JSON.parse(options.input);assert.equal(payload.bundleSha256,bundle.sha256);assert.equal(payload.port,8766);assert.ok(!command.includes(payload.bundle));assert.ok(!command.includes(token));return{code:installFailure?1:0,stdout:JSON.stringify({schema:1,started:true,reused:false,dataDir:'/home/synthetic/.local/share/tensorfold-cluster/state',service:'/home/synthetic/.config/systemd/user/tensorfold-cluster.service'}),stderr:'sensitive remote output must never be surfaced'}}
  if(command.includes('process.stdout.write(token)'))return{code:0,stdout:token,stderr:''}
  return{code:0,stdout:JSON.stringify(report),stderr:''}
 }}
 const setup=createClusterSshSetup({profileRoot:()=>profile,getSshHost:async()=>host,sessionForHost:async()=>session,agentBundle:async()=>bundle,now:()=>1000,newId:()=> 'synthetic-preview',delay:async()=>{},persistPreview:async record=>{if(persistFailure)throw new Error('Synthetic preview disk failure');records.set(record.preview.id,record)},loadPreview:async id=>records.get(id)??null,markApplied:async(_id,_command,state)=>{events.push('durable-'+state.state);states.push(state)},verifyAgent:async input=>{events.push('challenge');input.assertCurrent();assert.equal(input.session,session,'Credential proof remains on the captured verified SSH session');assert.equal(input.token,token);if(verificationFailure)throw new Error('Synthetic unsafe error '+token);return{protocol:1,nodeId:'synthetic-node',authenticated:!invalidProof as true}}})
 return{setup,records,states,calls,events,session,missingNode(){missingNode=true},report:(patch:any)=>Object.assign(report,patch),profile:(value:string)=>{profile=value},host:(patch:any)=>{host={...host,...patch}},failure:(kind:string)=>{if(kind==='persist')persistFailure=true;if(kind==='install')installFailure=true;if(kind==='verify')verificationFailure=true;if(kind==='proof')invalidProof=true}}
}
const commandId='synthetic-command-0001'
let h=fixture(),preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766})
assert.deepEqual(preview.blockers,[]);assert.equal(h.records.size,1);assert.ok(!JSON.stringify(preview).includes('profile'));assert.ok(!JSON.stringify(preview).includes(token));assert.equal(h.calls.filter(row=>row.options.input).length,0)
const applied=await h.setup.apply({previewId:preview.id,digest:preview.digest,commandId})
assert.match(applied.hostStamp,/^[a-f0-9]{64}$/);assert.equal(applied.verified,true);assert.equal(applied.token,token);assert.equal(applied.service,'systemd-user')
assert.ok(h.events.indexOf('durable-applying')<h.events.indexOf('remote-mutation'));assert.ok(h.events.indexOf('challenge')<h.events.indexOf('durable-verified'))
h=fixture();h.failure('persist');await assert.rejects(()=>h.setup.preview({sshHostId:'synthetic-host',port:8766}),/disk failure/);assert.equal(h.calls.filter(row=>row.options.input).length,0)
h=fixture();h.missingNode();preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});assert.equal(preview.nodeAvailable,false);assert.equal(preview.blockers.length,1);assert.match(preview.blockers[0],/Node.js 22/)
for(const patch of [{nodeVersion:'20.0.0'},{enginePath:null,engineVersion:null},{uid:0},{lingering:false},{serviceReady:false},{unsafeExisting:true}]){
 h=fixture();h.report(patch);preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});assert.ok(preview.blockers.length);await assert.rejects(()=>h.setup.apply({previewId:preview.id,digest:preview.digest,commandId}),/blocked/);assert.equal(h.calls.filter(row=>row.options.input).length,0)
}
for(const change of ['host','profile','report','digest']){
 h=fixture();preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});if(change==='host')h.host({updatedAt:2});if(change==='profile')h.profile('/synthetic/other-profile');if(change==='report')h.report({nodeVersion:'24.0.0'});
 await assert.rejects(()=>h.setup.apply({previewId:preview.id,digest:change==='digest'?'a'.repeat(64):preview.digest,commandId}));assert.equal(h.calls.filter(row=>row.options.input).length,0)
}
h=fixture();h.session.host.hostName='different-session.local';await assert.rejects(()=>h.setup.preview({sshHostId:'synthetic-host',port:8766}),/older or different/);assert.equal(h.calls.length,0)
h=fixture();preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});h.session.host={...h.session.host,hostName:'different-session.local'};await assert.rejects(()=>h.setup.apply({previewId:preview.id,digest:preview.digest,commandId}),/identity changed/);assert.equal(h.calls.filter(row=>row.options.input).length,0)
for(const fail of ['install','verify','proof']){
 h=fixture();preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});h.failure(fail)
 await assert.rejects(()=>h.setup.apply({previewId:preview.id,digest:preview.digest,commandId}),error=>{assert.ok(!String(error).includes(token));return true});assert.equal(h.states.at(-1)?.state,'needs-review');assert.ok(!h.states.some(row=>row.state==='verified'))
}
h=fixture();preview=await h.setup.preview({sshHostId:'synthetic-host',port:8766});const aborted=new AbortController();aborted.abort();await assert.rejects(()=>h.setup.apply({previewId:preview.id,digest:preview.digest,commandId,signal:aborted.signal}),/cancellation/);assert.equal(h.calls.filter(row=>row.options.input).length,0)
console.log('SSH setup: syntax-compiled remote scripts and fake SSH only, durable preview/start fences, blockers, unchanged host/profile/digest/prereqs, stdin transfer, captured-session proof, secret redaction and uncertain outcomes passed')
