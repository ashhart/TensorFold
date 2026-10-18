import assert from 'node:assert/strict'
import vm from 'node:vm'
import path from 'node:path'
import { constants } from 'node:fs'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { SSH_INSTALL_PROGRAM, SSH_CREDENTIAL_PROGRAM } from './ssh-install-script'
const home='/home/synthetic',root=home+'/.local/share/tensorfold-cluster',credential='synthetic_management_credential_'.repeat(3)
type Entry={kind:'directory'|'file'|'symlink';mode:number;uid:number;data:Buffer}
function fixture(platform='linux'){
 const files=new Map<string,Entry>([[home,{kind:'directory',mode:0o755,uid:1001,data:Buffer.alloc(0)}]]),fds=new Map<number,string>(),commands:any[]=[]
 let nextFd=1,uid=1001,failCommand=false
 const absent=()=>Object.assign(new Error('Synthetic missing file'),{code:'ENOENT'})
 const fs:any={constants,
 lstatSync(file:string){const e=files.get(file);if(!e)throw absent();return{...e,size:e.data.length,isDirectory:()=>e.kind==='directory',isFile:()=>e.kind==='file',isSymbolicLink:()=>e.kind==='symlink'}},
 statSync(file:string){return fs.lstatSync(file)},existsSync:(file:string)=>files.has(file),
 mkdirSync(file:string,{mode}:any){assert.ok(!files.has(file));files.set(file,{kind:'directory',mode,uid,data:Buffer.alloc(0)})},
 openSync(file:string,flags:number,mode:number){assert.ok(flags&constants.O_EXCL);assert.ok(flags&constants.O_NOFOLLOW);if(files.has(file))throw Object.assign(new Error('Synthetic existing file'),{code:'EEXIST'});assert.equal(mode,0o600);files.set(file,{kind:'file',mode,uid,data:Buffer.alloc(0)});const fd=nextFd++;fds.set(fd,file);return fd},
 writeFileSync(fd:number,data:any){files.get(fds.get(fd)!)!.data=Buffer.from(data)},fsyncSync(){},closeSync(fd:number){fds.delete(fd)},
 readFileSync(file:string,encoding?:string){const e=files.get(file);if(!e)throw absent();return encoding?e.data.toString():e.data},
 }
 const cp={execFileSync(command:string,args:any,options:any){commands.push({command,args,options});assert.equal(options.stdio,'ignore');if(failCommand)throw new Error('Synthetic unsafe supervisor error '+credential);return Buffer.alloc(0)}}
 const invoke=(program:string,payload?:unknown)=>{
  const stdin=new EventEmitter(),stdout:string[]=[],stderr:string[]=[],process:any={stdin,stdout:{write:(value:any)=>stdout.push(String(value))},stderr:{write:(value:any)=>stderr.push(String(value))},getuid:()=>uid,execPath:'/synthetic/bin/node',exitCode:0,exit(){throw new Error('Synthetic bounded input exit')}}
  new vm.Script(program).runInNewContext({require:(name:string)=>{if(name==='node:fs')return fs;if(name==='node:os')return{homedir:()=>home,platform:()=>platform};if(name==='node:path')return path;if(name==='node:child_process')return cp;if(name==='node:crypto')return{createHash};throw new Error('Unexpected remote program dependency')},process,Buffer})
  if(payload!==undefined){stdin.emit('data',Buffer.from(JSON.stringify(payload)));stdin.emit('end')}
  return{stdout:stdout.join(''),stderr:stderr.join(''),code:process.exitCode}
 }
 return{files,commands,invoke,setUid(value:number){uid=value},fail(){failCommand=true}}
}
const bytes=Buffer.from('module.exports={synthetic:true}'),payload={schema:1,home,uid:1001,platform:'linux',nodePath:'/synthetic/bin/node',enginePath:'/synthetic/bin/tensorfold-native',port:8766,commandId:'synthetic-command-0001',bundleSha256:createHash('sha256').update(bytes).digest('hex'),bundle:bytes.toString('base64')}
let h=fixture(),result=h.invoke(SSH_INSTALL_PROGRAM,payload)
assert.equal(result.code,0);assert.equal(JSON.parse(result.stdout).started,true);assert.equal(JSON.parse(result.stdout).verified,undefined,'A supervisor receipt alone never claims an authenticated live agent')
assert.equal(h.commands.length,2);assert.deepEqual(Array.from(h.commands[1].args),['--user','enable','--now','tensorfold-cluster.service'])
assert.equal(h.files.get(root)!.mode,0o700);assert.equal(h.files.get(root+'/config.json')!.mode,0o600);assert.equal(JSON.parse(h.files.get(root+'/config.json')!.data.toString()).host,'127.0.0.1')
assert.ok(![...h.files.values()].some(entry=>entry.data.toString().includes(credential)))
const original=[...h.files].map(([name,entry])=>[name,entry.data.toString()]);result=h.invoke(SSH_INSTALL_PROGRAM,payload);assert.equal(JSON.parse(result.stdout).reused,true);assert.equal(h.commands.length,2);assert.deepEqual([...h.files].map(([name,entry])=>[name,entry.data.toString()]),original)
result=h.invoke(SSH_INSTALL_PROGRAM,{...payload,bundleSha256:'a'.repeat(64)});assert.equal(result.code,1)
h=fixture('darwin');result=h.invoke(SSH_INSTALL_PROGRAM,{...payload,platform:'darwin'});assert.equal(result.code,0);assert.deepEqual(Array.from(h.commands[0].args).slice(0,2),['bootstrap','gui/1001']);assert.ok(h.files.get(home+'/Library/LaunchAgents/dev.tensorfold.cluster.plist')!.data.toString().includes('--config'))
h=fixture();h.files.set(home+'/.local',{kind:'symlink',mode:0o700,uid:1001,data:Buffer.alloc(0)});result=h.invoke(SSH_INSTALL_PROGRAM,payload);assert.equal(result.code,1);assert.equal(h.commands.length,0);assert.equal(h.files.has(root+'/cluster-node.js'),false)
h=fixture();h.setUid(0);result=h.invoke(SSH_INSTALL_PROGRAM,{...payload,uid:0});assert.equal(result.code,1);assert.equal(h.files.size,1)
h=fixture();h.fail();result=h.invoke(SSH_INSTALL_PROGRAM,payload);assert.equal(result.code,1);assert.ok(!result.stderr.includes(credential));assert.equal(h.files.has(root+'/intent.json'),true);assert.equal(h.files.has(root+'/install.json'),false,'An unsettled start preserves its intent and cannot masquerade as a completed install')
h=fixture();h.invoke(SSH_INSTALL_PROGRAM,payload);h.files.set(root+'/state/management.token',{kind:'file',mode:0o600,uid:1001,data:Buffer.from(credential)});result=h.invoke(SSH_CREDENTIAL_PROGRAM);assert.equal(result.stdout,credential)
h.files.get(root+'/state/management.token')!.mode=0o644;result=h.invoke(SSH_CREDENTIAL_PROGRAM);assert.equal(result.code,1);assert.equal(result.stdout,'');assert.ok(!result.stderr.includes(credential))
h.files.get(root+'/state/management.token')!.mode=0o600;h.files.get(root+'/state')!.kind='symlink';result=h.invoke(SSH_CREDENTIAL_PROGRAM);assert.equal(result.code,1);assert.equal(result.stdout,'')
console.log('Actual fixed remote install/credential programs with fake VM fs/process only: Linux/mac user service plans, exclusive paths, no root/overwrites, reuse, partial intent, loopback config and credential secrecy passed')
