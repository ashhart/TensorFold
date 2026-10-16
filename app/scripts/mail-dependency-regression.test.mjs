import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { createRequire } from 'node:module'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import os from 'node:os'

const root = path.resolve(import.meta.dirname, '..')
const require = createRequire(path.join(root, 'package.json'))
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-mail-parser-'))
try {
  const nodemailer = path.join(tmp, 'nodemailer.mjs')
  const network = path.join(tmp, 'network.mjs')
  const child = path.join(tmp, 'compose.mjs')
  // Keep the real installed MIME composer and actual approved-send consumer.
  // JSON transport substitutes external delivery; resolver substitution avoids DNS.
  await fs.writeFile(nodemailer, `import actual from ${JSON.stringify(require.resolve('nodemailer'))};export default {createTransport:()=>actual.createTransport({jsonTransport:true})};`, { mode: 0o600 })
  await fs.writeFile(network, `export const resolvePublicMailHost=async()=>({hostname:'smtp.example.test',addresses:[{address:'93.184.215.14',family:4}]});export const createPinnedSocketFactory=()=>()=>{};`, { mode: 0o600 })
  await fs.writeFile(child, `
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(${JSON.stringify(path.join(root, 'package.json'))});
const {createJiti}=require('jiti');
const jiti=createJiti(${JSON.stringify(path.join(root, 'package.json'))},{alias:{nodemailer:${JSON.stringify(nodemailer)},'./mail-network':${JSON.stringify(network)}},moduleCache:false,fsCache:false});
const {sendEmailOverSmtp}=await jiti.import(${JSON.stringify(path.join(root, 'electron/agent/smtp-send.ts'))});
const recipient=process.argv[2]==='malformed'?'[x]'.repeat(40000):'recipient@example.test';
const sent=await sendEmailOverSmtp({approvalStatus:'approved',payload:{to:recipient,subject:'subject',body:'body'}},{host:'smtp.example.test',user:'sender@example.test',password:'synthetic-only'});
assert.equal(sent.transport,'smtp');
assert.equal(sent.envelope.to,recipient);
console.log('composition returned');
`, { mode: 0o600 })
  for (const input of ['normal', 'malformed']) {
    const result = spawnSync(process.execPath, [child, input], { cwd: root, encoding: 'utf8', timeout: 3_000 })
    assert.equal(result.error, undefined, `${input} approved composition cannot stall the main process`)
    assert.equal(result.status, 0, result.stderr)
    assert.match(result.stdout, /composition returned/)
  }
  console.log('approved SMTP real MIME composition handles normal and bounded malformed recipients')
} finally {
  await fs.rm(tmp, { recursive: true, force: true })
}
