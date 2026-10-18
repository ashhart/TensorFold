import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as crypto from 'node:crypto'
import ts from 'typescript'
let directClosed=0,directCalls=0
let forwardCalls=0,closed=0,blocked=false,hold=false,release!:()=>void
const opened:any[]=[]
const source=readFileSync('electron/main/host/clusters/network.ts','utf8'),module={exports:{} as any}
class FakeClient {
 constructor(readonly options:any){}
 async identity(){await this.options.assertCurrent();return'node-a'}
}
const forward={async open(id:string,port:number,options:any){forwardCalls++;if(hold)await new Promise<void>(resolve=>{const before=release;release=()=>{before?.();resolve()}});options.assertCurrent?.();let disposed=false;const value={url:'http://127.0.0.1:19000',location:'remote',sshHostId:id,remotePort:port,async close(){if(!disposed){disposed=true;closed++}}};opened.push({options,value});return value}}
const imports:Record<string,any>={'node:crypto':crypto,'./client':{ClusterNodeClient:FakeClient},'./ssh-forward':{createClusterSshForward:()=>forward},'./ssh-http':{createSshHttpRequest:(session:any,port:number,current:any)=>{assert.ok(session);assert.equal(port,8766);directCalls++;return{request:async()=>{await current();return new Response('{}')},close(){directClosed++}}}}}
new Function('require','module','exports',ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText)((id:string)=>{assert.ok(id in imports,id);return imports[id]},module,module.exports)
const {createClusterNetwork,clusterSshStamp}=module.exports
let saved:any={id:'ssh-a',label:'Synthetic',hostName:'synthetic.local',port:22,identityFiles:[],origin:'manual',createdAt:1,updatedAt:1}
class Session { host=structuredClone(saved); listeners=new Set<()=>void>(); onClosed(listener:()=>void){this.listeners.add(listener);return()=>this.listeners.delete(listener)} close(){for(const listener of this.listeners)listener();this.listeners.clear()} }
let session:any=new Session();const host:any={id:'host-a',revision:1,token:'synthetic_management_credential_'.repeat(3),nodeId:'node-a',connection:{kind:'ssh',sshHostId:'ssh-a',port:8766},sshStamp:clusterSshStamp(saved)}
const network=createClusterNetwork({assertCurrent(){if(blocked)throw new Error('Synthetic profile changed')},hostCurrent:async()=>{},resolveSshHost:async()=>saved,sessionForHost:async()=>session})
const client=await network.client(host);assert.equal(forwardCalls,1);assert.equal(opened[0].options.session,session);await client.options.assertCurrent()
saved={...saved,updatedAt:2};await assert.rejects(()=>client.options.assertCurrent(),/changed/);await assert.rejects(()=>network.client(host),/target changed/);saved={...saved,updatedAt:1}
session.close();for(let i=0;i<10;i++)await Promise.resolve()
await assert.rejects(()=>client.options.assertCurrent(),/closed|changed|transport|session/i,'An old management client cannot release credentials after its exact SSH transport closes')
session=new Session();await network.client(host);assert.equal(forwardCalls,2,'Same-stamp reconnection must open a forward on the new SSH session, never reuse a stale loopback port')
await network.invalidate(host.id);assert.equal(closed,2)
await network.verify({sshHostId:'ssh-a',port:8766,token:host.token,session,assertCurrent(){}});assert.equal(closed,2);assert.equal(directCalls,1);assert.equal(directClosed,1)
await network.close();blocked=true;await assert.rejects(()=>network.client(host),/changed/);blocked=false
// Concurrent requests for one machine must share a pending forward and leave no orphan listener.
forwardCalls=0;closed=0;hold=true;release=undefined as any
const concurrent=createClusterNetwork({assertCurrent(){},hostCurrent:async()=>{},resolveSshHost:async()=>saved,sessionForHost:async()=>session})
const one=concurrent.client(host),two=concurrent.client(host)
for(let i=0;i<20;i++)await Promise.resolve()
assert.equal(forwardCalls,1,'Concurrent inspection/control requests must own exactly one forward per host revision')
release();await Promise.all([one,two]);await concurrent.close();assert.equal(closed,1,'Closing the network must own and close every opened forward')
hold=true;forwardCalls=0;closed=0;release=undefined as any
const closing=createClusterNetwork({assertCurrent(){},hostCurrent:async()=>{},resolveSshHost:async()=>saved,sessionForHost:async()=>session})
const pendingClient=closing.client(host).then(()=>null,(error:unknown)=>error)
for(let i=0;i<20;i++)await Promise.resolve()
let finished=false;const closingAll=closing.close().then(()=>{finished=true});await Promise.resolve();assert.equal(finished,false,'Network close waits for a pending owned forward to settle')
release();await closingAll;assert.ok(await pendingClient);assert.equal(finished,true)
console.log('Cluster network with fake forward/client: saved target/session binding, same-session enrollment proof, invalidation, profile closure and concurrent forward ownership passed')
