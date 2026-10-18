import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
const printer=ts.createPrinter()
function extract(file:string,names:string[]) {
 const ast=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.ES2022,true)
 return names.map(name=>{const node=ast.statements.find((row):row is ts.FunctionDeclaration=>ts.isFunctionDeclaration(row)&&row.name?.text===name);assert.ok(node,name);return printer.printNode(ts.EmitHint.Unspecified,node,ast).replace(/^export\s+/,'')}).join('\n')
}
const writeCode=extract('electron/store/write-infra.ts',['atomicWriteJson','writeJson'])
const settingsCode=extract('electron/store/persistence.ts',['saveSettings','writeModelsJson'])
let profile='/synthetic/one',allowed=true,failOn='',changeOn='',effect:()=>void=()=>{},writeCounter=0
const files=new Map<string,string>(),opened:string[]=[],renamed:string[]=[]
const originalSettings='{"sshHosts":[{"id":"original"}]}'
files.set('/synthetic/one/settings.json',originalSettings);files.set('/synthetic/two/settings.json','{"sshHosts":[{"id":"other-profile"}]}')
const io={async mkdir(){},async chmod(){},async unlink(file:string){files.delete(file);staged.delete(file)},async open(file:string){opened.push(file);let body='';return{async writeFile(value:string){body=value},async datasync(){if(file.includes(changeOn)&&changeOn)effect();if(file.includes(failOn)&&failOn)throw new Error('Synthetic fsync failure')},async close(){},get body(){return body}}},async rename(from:string,to:string){renamed.push(to);files.set(to,staged.get(from)!);staged.delete(from)}}
const staged=new Map<string,string>(),open=io.open
io.open=async file=>{const h=await open(file);return{...h,async writeFile(value:string){await h.writeFile(value);staged.set(file,value)}}}
const env:Record<string,any>={fs:io,path,process,writeChains:new Map(),quarantinedProtectedFiles:new Map(),writeCounter}
const writes=new Function(...Object.keys(env),ts.transpileModule(writeCode+'\nreturn {writeJson};',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText)(...Object.values(env))
const dependencies:Record<string,any>={...writes,settingsPath:()=>profile+'/settings.json',modelsPath:()=>profile+'/models.json',validateEngineModelProfileSets(){},protectSettings:(value:any)=>value,normalizeStoredUiPreferences:(value:any)=>value,normalizeStoredHelperDefaults:(value:any)=>value,getLegacySecretTemplate:()=>null,mergeUnresolvedLegacySecrets:(_before:any,next:any)=>next,countLegacySecrets:()=>0,setLegacySecretTemplate(){},setLegacySettingsBackupPath(){}}
const {saveSettings}=new Function(...Object.keys(dependencies),ts.transpileModule(settingsCode+'\nreturn {saveSettings};',{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText)(...Object.values(dependencies))
const data={providers:[{id:'synthetic-provider',models:['synthetic-model']}],sshHosts:[{id:'requested'}]}
const guard=()=>{if(profile!=='/synthetic/one')throw new Error('Synthetic profile changed');if(!allowed)throw new Error('Synthetic role revoked')}
failOn='settings.json';await assert.rejects(()=>saveSettings(data,{validateCommit:guard}),/fsync failure/);assert.equal(files.get('/synthetic/one/settings.json'),originalSettings);failOn=''
changeOn='models.json';effect=()=>{allowed=false};await assert.rejects(()=>saveSettings(data,{validateCommit:guard}),/role revoked/);assert.equal(files.get('/synthetic/one/settings.json'),originalSettings);allowed=true
changeOn='settings.json';effect=()=>{allowed=false};await assert.rejects(()=>saveSettings(data,{validateCommit:guard}),/role revoked/);assert.equal(files.get('/synthetic/one/settings.json'),originalSettings);allowed=true
changeOn='models.json';effect=()=>{profile='/synthetic/two'};await assert.rejects(()=>saveSettings(data,{validateCommit:guard}),/profile changed/);assert.equal(files.get('/synthetic/one/settings.json'),originalSettings);assert.equal(files.get('/synthetic/two/settings.json'),'{"sshHosts":[{"id":"other-profile"}]}');assert.ok(!opened.some(file=>file.startsWith('/synthetic/two/')));profile='/synthetic/one'
changeOn='';await saveSettings(data,{validateCommit:guard});assert.equal(JSON.parse(files.get('/synthetic/one/settings.json')!).sshHosts[0].id,'requested')
assert.ok(renamed.includes('/synthetic/one/models.json'));assert.ok(renamed.includes('/synthetic/one/settings.json'))
console.log('Actual saveSettings/writeJson/atomicWriteJson with fake fs: captured profile paths, catalogue and authoritative pre-rename guards, fsync faults and revoked-role/profile changes preserve prior settings passed')
