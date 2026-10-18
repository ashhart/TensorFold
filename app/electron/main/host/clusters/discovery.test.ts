import assert from 'node:assert/strict'
import { createClusterDiscovery, parseAvahiCandidates, parseBonjourResolution, CLUSTER_BONJOUR_SERVICE } from './discovery'
const good='=;eth0;IPv4;Synthetic node;_tensorfold-cluster._tcp;local;synthetic.local;192.168.1.4;8766;"protocol=1"'
const entries=parseAvahiCandidates([good,good.replace('protocol=1','protocol=2'),good.replace('synthetic.local','public.example.com'),good.replace('8766','70000'),good.replace('Synthetic node','bad\u202ename')].join('\n'),100)
assert.equal(entries.length,1);assert.equal(entries[0].trusted,false);assert.equal(entries[0].expiresAt,45100)
assert.equal(parseBonjourResolution('Synthetic','Synthetic can be reached at peer.local.:8766 (interface 2)\n protocol=1',200)?.host,'peer.local')
assert.equal(parseBonjourResolution('Synthetic','Synthetic can be reached at 169.254.169.254:80 (interface 2)\n protocol=1',200),null)
assert.equal(parseAvahiCandidates(Array.from({length:100},(_,n)=>good.replace('Synthetic node','Synthetic '+n)).join('\n'),0).length,32)
const commands:any[]=[]
const scanner=createClusterDiscovery({platform:'linux',now:()=>100,run:async(command,args,options)=>{commands.push({command,args,options});return{code:0,stdout:good,stderr:''}}})
assert.equal((await scanner.scan()).length,1);assert.deepEqual(commands[0].args,['--resolve','--parsable','--terminate',CLUSTER_BONJOUR_SERVICE]);assert.ok(commands[0].options.maxBytes<=65536)
let release!:()=>void,closed=false
const pendingScanner=createClusterDiscovery({platform:'linux',run:async(_command,_args,options)=>{await new Promise<void>(resolve=>{release=()=>{closed=true;resolve()}});assert.equal(options.signal.aborted,true);return{code:0,stdout:good,stderr:''}}})
const first=pendingScanner.scan(),duplicate=pendingScanner.scan();assert.equal(first,duplicate)
pendingScanner.cancel();let settled=false;void pendingScanner.settle().then(()=>settled=true);await Promise.resolve();assert.equal(settled,false,'Cancellation waits for injected physical command settlement')
release();assert.deepEqual(await first,[]);await pendingScanner.settle();assert.equal(closed,true)
let concurrent=0,max=0
const mac=createClusterDiscovery({platform:'darwin',now:()=>100,run:async(_command,args)=>{
 if(args[0]==='-B')return{code:0,stdout:Array.from({length:40},(_,i)=>`12:00 Add 2 4 local. _tensorfold-cluster._tcp. Node${i}`).join('\n'),stderr:''}
 concurrent++;max=Math.max(max,concurrent);await Promise.resolve();concurrent--;return{code:0,stdout:`${args[1]} can be reached at node.local.:8766 (interface 2)\n protocol=1`,stderr:''}
}})
assert.equal((await mac.scan()).length,16);assert.ok(max<=2)
console.log('Bonjour discovery: fake commands only, private/local candidates, protocol/port/TTL/bounds/control rejection, no credential/scan/enrollment, cancellation settlement and resolver concurrency passed')
