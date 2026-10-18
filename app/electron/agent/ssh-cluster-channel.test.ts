import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { SshSession } from './ssh-connection'
const host:any={id:'synthetic',label:'Synthetic',hostName:'synthetic.local',port:22,identityFiles:[],origin:'manual',createdAt:1,updatedAt:1}
class Client extends EventEmitter {
 calls:any[]=[]; pending:((error:Error|undefined,channel:any)=>void)|null=null; stdin:any[]=[]
 forwardOut(...args:any[]){this.calls.push(args.slice(0,4));this.pending=args[4]}
 exec(command:string,_options:any,callback:any){const stream:any=new EventEmitter();stream.stderr=new EventEmitter();stream.destroy=()=>{};stream.end=(input:any)=>{this.stdin.push(input);queueMicrotask(()=>{stream.emit('data',Buffer.from('synthetic result'));stream.emit('close',0,null)})};callback(undefined,stream)}
 end(){this.emit('close')}
}
const client=new Client(),session=new SshSession(host,client as any,()=>1)
await assert.rejects(()=>session.forwardLoopback(0),/valid/);assert.equal(client.calls.length,0)
host.hostName='mutated-after-session.local';host.identityFiles.push('/synthetic/new-key');assert.equal(session.host.hostName,'synthetic.local');assert.deepEqual(session.host.identityFiles,[]);assert.equal(Object.isFrozen(session.host),true)
const opening=session.forwardLoopback(8766),channel=new PassThrough();client.pending!(undefined,channel)
assert.equal(await opening,channel);assert.deepEqual(client.calls[0],['127.0.0.1',0,'127.0.0.1',8766])
const aborter=new AbortController(),late=session.forwardLoopback(8766,aborter.signal);aborter.abort();await assert.rejects(()=>late,/cancelled/)
const lateChannel=new PassThrough();client.pending!(undefined,lateChannel);assert.equal(lateChannel.destroyed,true)
const timed=session.forwardLoopback(8766,undefined,2);await assert.rejects(()=>timed,/timed out/);const timedChannel=new PassThrough();client.pending!(undefined,timedChannel);assert.equal(timedChannel.destroyed,true)
let closed=0;const remove=session.onClosed(()=>closed++);remove();session.onClosed(()=>closed++)
const result=await session.exec('fixed reviewed command',{input:'synthetic-input-not-in-command',timeoutMs:1000,maxBuffer:1024});assert.equal(result.code,0);assert.equal(client.stdin[0],'synthetic-input-not-in-command')
await assert.rejects(()=>session.exec('fixed reviewed command',{input:Buffer.alloc(8*1024*1024+1)}),/bounded transfer/);assert.equal(client.stdin.length,1)
client.end();assert.equal(closed,1);await assert.rejects(()=>session.forwardLoopback(8766),/unavailable/)
console.log('SSH cluster channel seam: fake clients only, loopback-only destination, invalid ports, cancellation/timeout late-channel destruction, close lifecycle and bounded stdin passed')
