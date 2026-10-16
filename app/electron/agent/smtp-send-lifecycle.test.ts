import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-smtp-lifecycle-'))
const state = { created: 0, closed: 0, reject: true }
const key = '__studioSmtpLifecycleFixture'
;(globalThis as any)[key] = state
try {
  const nodemailer = path.join(tmp, 'nodemailer.mjs')
  const network = path.join(tmp, 'network.mjs')
  await fs.writeFile(nodemailer, `export default {createTransport:()=>{const s=globalThis.${key};s.created++;return {sendMail:async()=>{if(s.reject)throw new Error('synthetic send rejection');return {messageId:'synthetic-message'}},close:()=>{s.closed++}}}};`, { mode: 0o600 })
  await fs.writeFile(network, `export const resolvePublicMailHost=async()=>({hostname:'smtp.example.test',addresses:[{address:'93.184.215.14',family:4}]});export const createPinnedSocketFactory=()=>()=>{};`, { mode: 0o600 })
  const jiti = createJiti(__filename, { alias: { nodemailer, './mail-network': network }, moduleCache: false, fsCache: false })
  const { sendEmailOverSmtp } = await jiti.import<any>(path.join(__dirname, 'smtp-send.ts'))
  const event = { approvalStatus: 'approved', payload: { to: 'recipient@example.test', subject: 'subject', body: 'body' } }
  const account = { host: 'smtp.example.test', user: 'sender@example.test', password: 'synthetic-only' }

  await assert.rejects(() => sendEmailOverSmtp(event, account), /synthetic send rejection/)
  assert.equal(state.created, 1, 'the failing operation owned a transport')
  assert.equal(state.closed, 1, 'the failing operation releases its owned transport')
  state.reject = false
  const sent = await sendEmailOverSmtp(event, account)
  assert.equal(sent.messageId, 'synthetic-message')
  assert.equal(state.closed, 2, 'successful send releases its transport once')
  await assert.rejects(() => sendEmailOverSmtp({ ...event, approvalStatus: 'pending' }, account), /must be approved/)
  assert.equal(state.created, 2, 'unapproved sends never create a transport')
  console.log('SMTP success, rejection cleanup, and approval controls passed')
} finally {
  delete (globalThis as any)[key]
  await fs.rm(tmp, { recursive: true, force: true })
}
