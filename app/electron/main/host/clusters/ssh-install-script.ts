/** Fixed remote programs receive reviewed data through SSH stdin, never shell interpolation. */
export const SSH_PREREQUISITE_PROGRAM = String.raw`
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process');
const home=os.homedir(),uid=process.getuid?.(),platform=os.platform(),root=path.join(home,'.local','share','tensorfold-cluster');
const run=(file,args)=>{try{return cp.execFileSync(file,args,{encoding:'utf8',timeout:4000,maxBuffer:32768,stdio:['ignore','pipe','ignore'],env:{...process.env,TENSORFOLD_NO_UPDATE_CHECK:'1'}}).trim()}catch{return null}};
let enginePath=run('/bin/sh',['-c','command -v tensorfold-native']),engineVersion=null;
if(enginePath&&enginePath.startsWith('/')&&!/[\r\n\0]/.test(enginePath))engineVersion=run(enginePath,['--version']);else enginePath=null;
let supervisor=null,serviceReady=false,lingering=false;
if(platform==='darwin'){supervisor='launchd-user';serviceReady=run('/bin/launchctl',['print','gui/'+uid])!==null}
if(platform==='linux'){supervisor='systemd-user';serviceReady=run('/bin/systemctl',['--user','show-environment'])!==null;lingering=run('/bin/loginctl',['show-user',String(uid),'-p','Linger','--value'])==='yes'}
const service=platform==='darwin'?path.join(home,'Library','LaunchAgents','dev.tensorfold.cluster.plist'):path.join(home,'.config','systemd','user','tensorfold-cluster.service');
let existing=null,unsafeExisting=false;
try{const s=fs.lstatSync(root);if(!s.isDirectory()||s.isSymbolicLink()||s.uid!==uid||(s.mode&63))unsafeExisting=true;else{const f=path.join(root,'install.json'),st=fs.lstatSync(f);if(!st.isFile()||st.isSymbolicLink()||st.uid!==uid||(st.mode&63)||st.size>16384)unsafeExisting=true;else existing=JSON.parse(fs.readFileSync(f,'utf8'))}}catch(e){if(e.code!=='ENOENT')unsafeExisting=true;else if(fs.existsSync(root))unsafeExisting=true}
const homeStat=fs.lstatSync(home);
process.stdout.write(JSON.stringify({platform,arch:os.arch(),home,uid,nodePath:process.execPath,nodeVersion:process.versions.node,enginePath,engineVersion,supervisor,serviceReady,lingering,homeOwned:homeStat.isDirectory()&&!homeStat.isSymbolicLink()&&homeStat.uid===uid,serviceExists:fs.existsSync(service),existing,unsafeExisting}));
`

export const SSH_INSTALL_PROGRAM = String.raw`
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
let chunks=[],size=0;process.stdin.on('data',b=>{size+=b.length;if(size>8*1024*1024)process.exit(2);chunks.push(b)});
process.stdin.on('end',()=>{try{
const p=JSON.parse(Buffer.concat(chunks).toString('utf8')),uid=process.getuid?.(),home=os.homedir(),platform=os.platform();
if(uid===0||p.schema!==1||p.home!==home||p.uid!==uid||p.platform!==platform||p.nodePath!==process.execPath||!['darwin','linux'].includes(platform)||!Number.isInteger(p.port)||p.port<1||p.port>65535||!/^[A-Za-z0-9_-]{16,100}$/.test(p.commandId)||!/^[a-f0-9]{64}$/.test(p.bundleSha256)||typeof p.bundle!=='string'||typeof p.enginePath!=='string'||!path.isAbsolute(p.enginePath)||/[\0\r\n]/.test(p.enginePath))throw Error('Invalid reviewed installation.');
const bytes=Buffer.from(p.bundle,'base64');if(bytes.length>4*1024*1024||crypto.createHash('sha256').update(bytes).digest('hex')!==p.bundleSha256)throw Error('Agent bundle integrity mismatch.');
const root=path.join(home,'.local','share','tensorfold-cluster'),dataDir=path.join(root,'state'),entry=path.join(root,'cluster-node.js'),config=path.join(root,'config.json');
const service=platform==='darwin'?path.join(home,'Library','LaunchAgents','dev.tensorfold.cluster.plist'):path.join(home,'.config','systemd','user','tensorfold-cluster.service');
const checked=(file,directory,privateMode)=>{const s=fs.lstatSync(file);if(s.isSymbolicLink()||(directory?!s.isDirectory():!s.isFile())||s.uid!==uid||(s.mode&(privateMode?63:18)))throw Error('Unsafe installation path; no existing files were replaced.');return s};
const mkdir=dir=>{if(dir===home){checked(home,true,false);return}mkdir(path.dirname(dir));try{checked(dir,true,dir===root||dir.startsWith(root+path.sep))}catch(e){if(e.code!=='ENOENT')throw e;fs.mkdirSync(dir,{mode:448});checked(dir,true,dir===root||dir.startsWith(root+path.sep))}};
const write=(file,text)=>{const fd=fs.openSync(file,fs.constants.O_WRONLY|fs.constants.O_CREAT|fs.constants.O_EXCL|(fs.constants.O_NOFOLLOW||0),384);try{fs.writeFileSync(fd,text);fs.fsyncSync(fd)}finally{fs.closeSync(fd)}};
const marker=path.join(root,'install.json');
if(fs.existsSync(root)){
checked(root,true,true);const st=checked(marker,false,true);if(st.size>16384)throw Error('Installation marker exceeds its limit.');const old=JSON.parse(fs.readFileSync(marker,'utf8'));
if(old.schema!==1||old.bundleSha256!==p.bundleSha256||old.port!==p.port||old.enginePath!==p.enginePath||old.nodePath!==p.nodePath||old.state!=='started')throw Error('Existing installation needs deliberate reconciliation or upgrade.');
checked(entry,false,true);checked(config,false,true);checked(dataDir,true,true);checked(service,false,true);
if(crypto.createHash('sha256').update(fs.readFileSync(entry)).digest('hex')!==p.bundleSha256)throw Error('Existing agent bundle changed.');
process.stdout.write(JSON.stringify({schema:1,started:true,reused:true,dataDir,service}));return;
}
mkdir(root);mkdir(dataDir);mkdir(path.dirname(service));
write(path.join(root,'intent.json'),JSON.stringify({schema:1,commandId:p.commandId,bundleSha256:p.bundleSha256}));
write(entry,bytes);write(config,JSON.stringify({schema:1,dataDir,enginePath:p.enginePath,host:'127.0.0.1',port:p.port,advertise:false}));
const xml=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&apos;');
const quote=s=>'"'+String(s).replace(/\\/g,'\\\\').replace(/"/g,'\\"').replace(/%/g,'%%').replace(/\$/g,'$$$$')+'"';
if(platform==='darwin'){
write(path.join(root,'stdout.log'),'');write(path.join(root,'stderr.log'),'');
write(service,'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>Label</key><string>dev.tensorfold.cluster</string><key>ProgramArguments</key><array>'+[p.nodePath,entry,'--config',config].map(x=>'<string>'+xml(x)+'</string>').join('')+'</array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>WorkingDirectory</key><string>'+xml(root)+'</string><key>StandardOutPath</key><string>'+xml(path.join(root,'stdout.log'))+'</string><key>StandardErrorPath</key><string>'+xml(path.join(root,'stderr.log'))+'</string></dict></plist>');
cp.execFileSync('/bin/launchctl',['bootstrap','gui/'+uid,service],{stdio:'ignore',timeout:15000});
}else{
write(service,'[Unit]\nDescription=TensorFold managed cluster node\n[Service]\nType=simple\nExecStart='+[p.nodePath,entry,'--config',config].map(quote).join(' ')+'\nWorkingDirectory='+quote(root)+'\nUMask=0077\nRestart=on-failure\nRestartSec=2\n[Install]\nWantedBy=default.target\n');
cp.execFileSync('/bin/systemctl',['--user','daemon-reload'],{stdio:'ignore',timeout:10000});cp.execFileSync('/bin/systemctl',['--user','enable','--now','tensorfold-cluster.service'],{stdio:'ignore',timeout:15000});
}
write(marker,JSON.stringify({schema:1,commandId:p.commandId,bundleSha256:p.bundleSha256,port:p.port,enginePath:p.enginePath,nodePath:p.nodePath,state:'started'}));
process.stdout.write(JSON.stringify({schema:1,started:true,reused:false,dataDir,service}));
}catch{process.stderr.write('Reviewed per-user setup did not settle; inspect its existing files and service before retrying.');process.exitCode=1}});
`

/** Dedicated credential read: output stays in this main-process setup adapter, never renderer/logs. */
export const SSH_CREDENTIAL_PROGRAM = String.raw`
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
try{const root=path.join(os.homedir(),'.local','share','tensorfold-cluster'),dir=path.join(root,'state'),file=path.join(dir,'management.token'),uid=process.getuid?.();for(const p of [root,dir,file]){const s=fs.lstatSync(p);if(s.isSymbolicLink()||s.uid!==uid||(s.mode&63)||(p===file?!s.isFile():!s.isDirectory()))throw Error('Unsafe credential path.')}if(fs.statSync(file).size>256)throw Error('Invalid credential size.');const token=fs.readFileSync(file,'utf8').trim();if(!/^[A-Za-z0-9_-]{43,160}$/.test(token))throw Error('Invalid credential.');process.stdout.write(token)}catch{process.stderr.write('Management credential is not ready.');process.exitCode=1}
`
