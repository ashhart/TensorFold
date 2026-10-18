import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
const handlers=new Map<string,(...args:any[])=>any>(),original:any={providers:[],sshHosts:[],unrelated:'preserve'}
const store:any={settings:original};let profile='/synthetic/one',role=true,fail=false,hold=false,release!:()=>void,leases=0,disconnected=0,trust=0,secrets=0,disk=structuredClone(original)
const manager={listAllHosts:async()=>store.settings.sshHosts,status:(id:string)=>({hostId:id}),disconnect(){disconnected++},test:async()=>({ok:true}),ensureConnected:async()=>{},resolveSecret(){secrets++},resolveTrust(){trust++},listRemoteDir:async()=>[]}
const imports:Record<string,unknown>={
 '../../../agent/security':{auditPreview:(value:any)=>String(value)},
 '../../../store':{vontraHomeDir:()=>profile,saveSettings:async(next:any,options:any)=>{assert.equal(leases,1);options.validateCommit();if(hold)await new Promise<void>(done=>{release=done});options.validateCommit();if(fail)throw new Error('Synthetic protected write failed');disk=structuredClone(next)}},
 '../../state':{store},'../audit':{auditEvent:async()=>{},requireSecurityRole(){if(!role)throw new Error('Synthetic role denied')}},
 '../ipc':{handle:(name:string,fn:any)=>handlers.set(name,fn)},
}
const module={exports:{} as any}
new Function('require','module','exports',ts.transpileModule(readFileSync('electron/main/host/ipc/ssh-ipc.ts','utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText)((id:string)=>{assert.ok(id in imports,id);return imports[id]},module,module.exports)
module.exports.registerSshIpc({sshManager:manager,profileRoot:()=>profile,reserveOperation(){leases++;return{release(){leases--}}}})
const call=(name:string,...args:any[])=>Promise.resolve().then(()=>handlers.get('ssh:'+name)!({},...args))
const host={id:'synthetic-host',label:'Synthetic host',hostName:'synthetic.local',port:22,identityFiles:[],origin:'manual',createdAt:1,updatedAt:1}
await assert.rejects(()=>call('saveHost',{...host,authToken:'never accepted'}),/Unsupported/)
await assert.rejects(()=>call('saveHost',{...host,port:0}),/valid/)
await assert.rejects(()=>call('saveHost',{...host,hostName:'name; unexpected command'}),/valid/)
fail=true;await assert.rejects(()=>call('saveHost',host),/write failed/);assert.equal(store.settings,original);assert.deepEqual(disk,original);fail=false
hold=true;const saving=call('saveHost',host);for(let i=0;i<10&&!release;i++)await Promise.resolve();assert.equal(store.settings,original,'Hosts are never published before disk commit');assert.equal(leases,1);release();await saving;hold=false
assert.equal(store.settings.sshHosts.length,1);assert.equal(store.settings.unrelated,'preserve');assert.equal(leases,0);assert.equal(trust,0,'Saving a host never trusts its host key')
const saved=store.settings,priorDisk=structuredClone(disk)
fail=true;await assert.rejects(()=>call('removeHost','synthetic-host'),/write failed/);assert.equal(store.settings,saved);assert.deepEqual(disk,priorDisk);assert.equal(disconnected,0);fail=false
hold=true;release=undefined as any;const removing=call('removeHost','synthetic-host');for(let i=0;i<10&&!release;i++)await Promise.resolve();role=false;release();await assert.rejects(()=>removing,/denied/);assert.equal(store.settings,saved);assert.deepEqual(disk,priorDisk);assert.equal(disconnected,0);role=true;hold=false
hold=true;release=undefined as any;const changing=call('saveHost',{...host,updatedAt:saved.sshHosts[0].updatedAt,label:'Changed'});for(let i=0;i<10&&!release;i++)await Promise.resolve();profile='/synthetic/two';release();await assert.rejects(()=>changing,/profile changed|settings or profile changed/i);assert.equal(store.settings,saved);assert.deepEqual(disk,priorDisk);profile='/synthetic/one';hold=false
await assert.rejects(()=>call('saveHost',{...host,updatedAt:0}),/changed/)
role=false;await assert.rejects(()=>call('resolveTrust','synthetic-prompt',true),/denied/);await assert.rejects(()=>call('resolveSecret','synthetic-prompt','synthetic'),/denied/);assert.equal(trust,0);assert.equal(secrets,0);role=true
await assert.rejects(()=>call('resolveTrust','synthetic-prompt','false'),/explicit/)
await call('resolveTrust','synthetic-prompt',false);assert.equal(trust,1)
await call('resolveSecret','synthetic-prompt',null);assert.equal(secrets,1)
await call('removeHost','synthetic-host');assert.equal(disconnected,1);assert.deepEqual(store.settings.sshHosts,[]);assert.equal(leases,0)
console.log('Actual SSH IPC: strict schemas, no auto-trust, awaited save/remove commit, save failure and mid-await role/profile/stale fences preserve prior host state; prompt roles and physical operation leases passed')
