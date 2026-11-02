import assert from 'node:assert/strict'
import { bindGmailInboxRun } from './gmail-inbox-run'
import { gmailInboxKey } from '../../../shared/gmail-inbox'

let thread: any = { id: 'task', kind: 'task', status: 'idle', providerId: 'local', workspace: null }
let provider: any = { id: 'tensorfold', kind: 'local', source: 'tensorfold', transport: 'chat-completions' }
let authorized = true, selected = true, research = false, saves = 0, failSave = false
const payload = { threadId: 'task', userText: 'Synthetic private mail', sourceThreadKey: gmailInboxKey('account-a', 1, 'dd11') }
const deps = {
  authorize() { if (!authorized) throw new Error('Owner required') },
  selected: async () => { if (!selected) throw new Error('Selected account changed') },
  thread: () => thread, provider: () => provider, externalResearch: () => research,
  persist: async () => { saves++; assert.equal(thread.googleLocalOnly, true); if (failSave) throw new Error('Protected persistence failed') },
}
authorized = false
await assert.rejects(bindGmailInboxRun(payload, deps), /Owner/); assert.equal(saves, 0)
authorized = true; selected = false
await assert.rejects(bindGmailInboxRun(payload, deps), /account changed/); assert.equal(saves, 0)
selected = true
for (const external of [{ id: 'cloud', kind: 'cloud', baseUrl: 'https://model.example.com' }, { id: 'lan', kind: 'local', baseUrl: 'http://192.0.2.20:8080' }, { id: 'cli', kind: 'local', transport: 'codex-cli' }]) {
  provider = external
  await assert.rejects(bindGmailInboxRun(payload, deps), /private Google account data/)
  assert.equal(thread.googleLocalOnly, undefined); assert.equal(saves, 0)
}
provider = { id: 'tensorfold', kind: 'local', source: 'tensorfold', transport: 'chat-completions' }
research = true
await assert.rejects(bindGmailInboxRun(payload, deps), /external research off/)
research = false; failSave = true
await assert.rejects(bindGmailInboxRun(payload, deps), /persistence failed/)
assert.equal(thread.googleLocalOnly, true, 'Persistence failure cannot reopen an external route in memory')
failSave = false
await bindGmailInboxRun(payload, deps)
assert.equal(saves, 2)
const before = saves
await bindGmailInboxRun({ threadId: 'task', userText: 'Ordinary text' }, deps)
assert.equal(saves, before, 'Ordinary task starts retain their existing behavior')
thread = { ...thread, googleLocalOnly: false, workspace: { kind: 'ssh' } }
await assert.rejects(bindGmailInboxRun(payload, deps), /private Google account data/)
console.log('Gmail task handoff enforces owner, source, local API route and durable monotonic privacy marking before inference.')
