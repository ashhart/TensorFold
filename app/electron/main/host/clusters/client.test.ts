import assert from 'node:assert/strict'
import { ClusterNodeClient, ClusterRemoteError } from './client'
const token='synthetic_management_credential_'.repeat(3)
const profile={id:'single',title:'Synthetic single profile',mode:'single',platform:'darwin',arch:'arm64',ranks:1,transports:['tcp'],modelTypes:['qwen3_5'],poolsWeights:false,requiresFabric:false,engineVersion:'1.0.12'}
const snapshot={protocol:1,nodeId:'node-a',version:'1',platform:'darwin',arch:'arm64',observedAt:1,totalMemoryBytes:10000,freeMemoryBytes:8000,engine:{available:true,version:'1.0.12',profiles:[profile],reason:null},models:[],links:[],active:null}
const response=(value:unknown,status=200)=>new Response(JSON.stringify(value),{status,headers:{'content-type':'application/json'}})
let calls=0,revoked=false
let client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:'node-a',assertCurrent(){if(revoked)throw new Error('Synthetic scope revoked')},request:async(url,init)=>{calls++;assert.equal(new Headers(init.headers).get('Authorization'),'Bearer '+token);assert.equal(init.redirect,'manual');if(url.includes('/identity?'))return response({protocol:1,nodeId:'node-a',nonce:new URL(url).searchParams.get('nonce'),authenticated:true});return response(snapshot)}})
assert.equal(await client.identity(),'node-a');assert.equal((await client.status()).nodeId,'node-a')
const before=calls;revoked=true;await assert.rejects(()=>client.status(),/revoked/);assert.equal(calls,before,'A revoked scope refuses before credential dispatch');revoked=false
client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:'node-a',assertCurrent(){},request:async()=>response({...snapshot,nodeId:'different-node'})})
await assert.rejects(()=>client.status(),error=>error instanceof ClusterRemoteError&&!error.uncertain)
client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:'node-a',assertCurrent(){},request:async()=>response({protocol:1,nodeId:'node-a',nonce:'wrong-nonce',authenticated:true})})
await assert.rejects(()=>client.identity(),/challenge/)
for(const code of [302,400,500]){
 calls=0;client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:null,assertCurrent(){},request:async()=>{calls++;return response({unsafe:'never expose '+token},code)}})
 await assert.rejects(()=>client.request('/api/cluster/v1/models/ensure',{commandId:'synthetic-command'}),error=>{assert.ok(error instanceof ClusterRemoteError);assert.equal(error.uncertain,code===302||code>=500);assert.ok(!String(error).includes(token));return true});assert.equal(calls,1,'Mutations never retry or follow redirects automatically')
}
calls=0;client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:null,assertCurrent(){},request:async()=>{calls++;throw new Error('Unsafe remote error '+token)}})
await assert.rejects(()=>client.request('/api/cluster/v1/models/ensure',{commandId:'synthetic-command'}),error=>error instanceof ClusterRemoteError&&error.uncertain&&!String(error).includes(token));assert.equal(calls,1)
let cancelled=false
client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:null,assertCurrent(){},request:async()=>new Response(new ReadableStream({start(controller){controller.enqueue(new Uint8Array(2_000_001))},cancel(){cancelled=true}}))})
await assert.rejects(()=>client.request('/api/cluster/v1/status'),/size limit/);assert.equal(cancelled,true)
client=new ClusterNodeClient({baseUrl:'https://synthetic.example.invalid',token,nodeId:null,assertCurrent(){},request:async()=>new Response('{"incomplete":')})
await assert.rejects(()=>client.request('/api/cluster/v1/models/ensure',{}),error=>error instanceof ClusterRemoteError&&error.uncertain)
console.log('Cluster client with fake HTTP responses: bearer/nonce/wrong-node/scope checks, no redirect/retry, malformed/oversized stream cancellation and uncertain mutation secrecy passed')
