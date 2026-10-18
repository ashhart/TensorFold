import assert from 'node:assert/strict'
import { constants } from 'node:fs'
import { createManagedSshKeyStore, ed25519PublicKey, type ClusterKeyFs } from './ssh-key'
const field = (value: string | Buffer) => { const b = Buffer.from(value), n = Buffer.alloc(4); n.writeUInt32BE(b.length); return Buffer.concat([n,b]) }
const pub = Buffer.alloc(32, 17), publicKey = ed25519PublicKey(pub)
const outer = Buffer.from(publicKey.split(' ')[1], 'base64'), one = Buffer.from([0,0,0,1])
let body = Buffer.concat([Buffer.alloc(8, 7),field('ssh-ed25519'),field(pub),field(Buffer.concat([Buffer.alloc(32,29),pub])),field('synthetic nonfunctional fixture')])
body = Buffer.concat([body,Buffer.from(Array.from({length:8-body.length%8},(_,i)=>i+1))])
const privateKey = '-----BEGIN OPENSSH PRIVATE KEY-----\n'+Buffer.concat([Buffer.from('openssh-key-v1\0'),field('none'),field('none'),field(''),one,field(outer),field(body)]).toString('base64')+'\n-----END OPENSSH PRIVATE KEY-----\n'
type Entry = { kind: 'directory'|'file'|'symlink'; mode: number; uid: number; data: Buffer }
const entries = new Map<string,Entry>([['/profile',{kind:'directory',mode:0o700,uid:1,data:Buffer.alloc(0)}]])
const error = (code: string) => Object.assign(new Error('Synthetic filesystem refusal.'),{code})
let profile = '/profile', generated = 0, failPublic = false
const io: ClusterKeyFs = {
 async lstat(file) { const e=entries.get(file); if(!e)throw error('ENOENT'); return {isDirectory:()=>e.kind==='directory',isFile:()=>e.kind==='file',isSymbolicLink:()=>e.kind==='symlink',mode:e.mode,uid:e.uid,size:e.data.length} },
 async mkdir(file,{mode}) { if(entries.has(file))throw error('EEXIST'); entries.set(file,{kind:'directory',mode,uid:1,data:Buffer.alloc(0)}) },
 async readFile(file) { const e=entries.get(file);if(!e)throw error('ENOENT');return e.data },
 async open(file,flags,mode) { assert.ok(flags&constants.O_EXCL);assert.ok(flags&constants.O_NOFOLLOW);if(entries.has(file))throw error('EEXIST');if(failPublic&&file.endsWith('.pub'))throw error('EIO');const entry={kind:'file' as const,mode,uid:1,data:Buffer.alloc(0)};entries.set(file,entry);return {async writeFile(value){entry.data=Buffer.from(value)},async sync(){},async close(){}} },
}
const store=createManagedSshKeyStore({profileRoot:()=>profile,fs:io,uid:1,generatePair:()=>{generated++;return{privateKey,publicKey:publicKey+'\n'}}})
assert.equal((await store.inspect()).available,false);assert.equal(generated,0)
const created=await store.create();assert.equal(created.available,true);assert.equal(created.publicKey,publicKey);assert.match(created.fingerprint!,/^SHA256:/)
assert.deepEqual(Object.keys(created).sort(),['available','fingerprint','message','publicKey']);assert.ok(!JSON.stringify(created).includes('PRIVATE KEY'))
assert.equal(entries.get('/profile/clusters/ssh/id_ed25519')!.mode,0o600);assert.equal(entries.get('/profile/clusters/ssh')!.mode,0o700)
await store.create();assert.equal(generated,1,'Existing valid keys are inspected and never overwritten or regenerated')
entries.get('/profile/clusters/ssh/id_ed25519')!.mode=0o644;await assert.rejects(()=>store.inspect(),/private/);entries.get('/profile/clusters/ssh/id_ed25519')!.mode=0o600
entries.get('/profile/clusters/ssh')!.kind='symlink';await assert.rejects(()=>store.create(),/symbolic/);entries.get('/profile/clusters/ssh')!.kind='directory'
entries.delete('/profile/clusters/ssh/id_ed25519.pub');await assert.rejects(()=>store.create(),/partial/);assert.equal(generated,1)
entries.delete('/profile/clusters/ssh/id_ed25519');failPublic=true;await assert.rejects(()=>store.create(),/Synthetic/);assert.equal(entries.has('/profile/clusters/ssh/id_ed25519'),true,'A partial pair is kept for explicit repair, never removed or overwritten behind the user')
await assert.rejects(()=>store.create(),/partial/)
console.log('Managed SSH key store: fake encoding/filesystem only, no key generation; exclusive private writes, public-only projection, permissions/symlink/partial-pair and no-overwrite guards passed')
