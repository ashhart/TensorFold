import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { createClusterSshForward } from './ssh-forward'
class FakeServer extends EventEmitter {
 listening=false;callback:any;bound:any;closed=0
 constructor(callback:any){super();this.callback=callback}
 listen(bound:any){this.bound=bound;queueMicrotask(()=>{this.listening=true;this.emit('listening')});return this}
 address(){return this.listening?{address:'127.0.0.1',port:19000,family:'IPv4'}:null}
 close(callback:any){this.listening=false;this.closed++;queueMicrotask(callback);return this}
}
class FakeSocket extends PassThrough { setTimeout(_time:number,_callback:any){return this} }
let server:FakeServer,hostClosed=()=>{},channels:PassThrough[]=[],opens=0
const session={async forwardLoopback(port:number){assert.equal(port,8766);opens++;const stream=new PassThrough();channels.push(stream);return stream},onClosed(listener:()=>void){hostClosed=listener;return()=>{hostClosed=()=>{}}}}
const factory=createClusterSshForward({sessionForHost:async()=>session,createServer:((callback:any)=>server=new FakeServer(callback)) as any,maxConnections:1})
const forward=await factory.open('synthetic-host',8766)
assert.deepEqual(server!.bound,{host:'127.0.0.1',port:0,exclusive:true});assert.equal(forward.location,'remote');assert.equal(forward.url,'http://127.0.0.1:19000')
const socket=new FakeSocket();server!.callback(socket);for(let i=0;i<6;i++)await Promise.resolve();assert.equal(opens,1)
const extra=new FakeSocket();server!.callback(extra);assert.equal(extra.destroyed,true,'Forward sockets remain bounded')
hostClosed();await forward.close();assert.equal(socket.destroyed,true);assert.equal(channels[0].destroyed,true);assert.equal(server!.listening,false)
await forward.close();assert.equal(server!.closed,1)
let lookedUp=0;const pinnedFactory=createClusterSshForward({sessionForHost:async()=>{lookedUp++;throw new Error('Must not rebind saved host during credential proof')},createServer:((callback:any)=>server=new FakeServer(callback)) as any})
const pinned=await pinnedFactory.open('synthetic-host',8766,{session});assert.equal(lookedUp,0);await pinned.close()
let checks=0
await assert.rejects(()=>factory.open('synthetic-host',8766,{assertCurrent(){if(++checks>1)throw new Error('Synthetic host changed')}}),/changed/)
console.log('SSH forward lifecycle: fake listener/sockets only, loopback binding, explicit remote provenance, connection bounds, channel/transport close and captured-session reuse passed')
