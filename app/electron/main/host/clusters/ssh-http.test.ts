import assert from 'node:assert/strict'
import { Duplex } from 'node:stream'
import { createSshHttpRequest } from './ssh-http'
class FakeHttpChannel extends Duplex {
 sent='';answered=false
 _read(){}
 _write(chunk:any,_encoding:any,callback:any){this.sent+=chunk.toString();callback();if(!this.answered&&this.sent.includes('\r\n\r\n')){this.answered=true;queueMicrotask(()=>{this.push('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{"ok":true}');this.push(null)})}}
 setTimeout(){return this} setNoDelay(){return this} setKeepAlive(){return this}
}
let closed=false,guard=true,opens=0,listener=()=>{},last:FakeHttpChannel
const session={onClosed(callback:()=>void){listener=callback;return()=>{}},async forwardLoopback(port:number){assert.equal(port,8766);opens++;last=new FakeHttpChannel();return last}}
const transport=createSshHttpRequest(session,8766,()=>{if(!guard)throw new Error('Synthetic owner changed')})
const response=await transport.request('http://127.0.0.1:19000/api/cluster/v1/status',{method:'GET',headers:{Authorization:'Bearer synthetic-management'}})
assert.deepEqual(await response.json(),{ok:true});assert.equal(opens,1);assert.match(last!.sent,/GET \/api\/cluster\/v1\/status HTTP\/1.1/);assert.match(last!.sent,/authorization: Bearer synthetic-management/i)
await assert.rejects(()=>transport.request('https://public.example.invalid/api/cluster/v1/status',{method:'GET'}),/Unsupported/);assert.equal(opens,1)
guard=false;await assert.rejects(()=>transport.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET'}),/owner changed/);assert.equal(opens,1);guard=true
listener();await assert.rejects(()=>transport.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET'}),/session closed/);assert.equal(opens,1);transport.close()
const aborted=new AbortController();aborted.abort()
const blockedSession={onClosed(){return()=>{}},async forwardLoopback(){throw new Error('No channel should open after abort')}}
const cancelled=createSshHttpRequest(blockedSession,8766,()=>{})
await assert.rejects(()=>cancelled.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET',signal:aborted.signal}));cancelled.close()
class ControlledChannel extends Duplex {
 sent=''
 _read(){}
 _write(chunk:any,_encoding:any,callback:any){this.sent+=chunk.toString();callback()}
 setTimeout(){return this} setNoDelay(){return this} setKeepAlive(){return this}
}
const waitFor=async(condition:()=>boolean)=>{for(let i=0;i<100&&!condition();i++)await new Promise<void>(resolve=>setImmediate(resolve));assert.ok(condition(),'The fake HTTP operation must physically settle')}
let channel!:ControlledChannel
const controlledSession={onClosed(){return()=>{}},async forwardLoopback(){channel=new ControlledChannel();return channel}}
const controlled=createSshHttpRequest(controlledSession,8766,()=>{})
for(const init of [{method:'DELETE'},{method:'POST',body:new URLSearchParams({secret:'synthetic'})}])await assert.rejects(()=>controlled.request('http://127.0.0.1/api/cluster/v1/status',init),/Unsupported/)
await assert.rejects(()=>controlled.request('http://127.0.0.1/v1/chat/completions',{method:'POST',body:'界'.repeat(350_000)}),/size limit/,'The request cap measures UTF-8 bytes rather than JavaScript character count')
const beforeHeaders=new AbortController()
const waiting=controlled.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET',signal:beforeHeaders.signal})
await waitFor(()=>!!channel?.sent);beforeHeaders.abort();await assert.rejects(()=>waiting,/settle|cancel/);assert.equal(channel.destroyed,true)
const afterHeaders=new AbortController()
channel=undefined as any
const streaming=controlled.request('http://127.0.0.1/v1/chat/completions',{method:'POST',body:'{"stream":true}',signal:afterHeaders.signal})
await waitFor(()=>!!channel?.sent);channel.push('HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 50\r\n\r\ndata: partial\n\n')
const streamingResponse=await streaming,body=streamingResponse.text();afterHeaders.abort();await assert.rejects(()=>body);assert.equal(channel.destroyed,true)
channel=undefined as any
const partial=controlled.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET'})
await waitFor(()=>!!channel?.sent);channel.push('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 80\r\n\r\n{"ok":');channel.push(null)
const partialResponse=await partial;await assert.rejects(()=>partialResponse.text(),'Truncated bodies must never become a complete management result');controlled.close()
let releaseChannel!:(value:ControlledChannel)=>void,scope=true,scopeOpens=0
const delayedSession={onClosed(){return()=>{}},async forwardLoopback(){scopeOpens++;return new Promise<ControlledChannel>(resolve=>{releaseChannel=resolve})}}
const revoked=createSshHttpRequest(delayedSession,8766,()=>{if(!scope)throw new Error('Synthetic scope changed')})
const late=revoked.request('http://127.0.0.1/api/cluster/v1/status',{method:'GET'});await waitFor(()=>scopeOpens===1);scope=false;const lateChannel=new ControlledChannel();releaseChannel(lateChannel);await assert.rejects(()=>late,/scope changed/);assert.equal(lateChannel.destroyed,true);assert.equal(lateChannel.sent,'','No credentials may be written after owner/profile scope changes during channel establishment');revoked.close()
console.log('Actual HTTP parser over fake SSH Duplex: verified destination/framing, bounded shape, scope drift, cancellation before and after headers, truncated body refusal and no network listener passed')
