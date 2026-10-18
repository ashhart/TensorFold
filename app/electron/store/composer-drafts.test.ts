import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync, mkdirSync, symlinkSync, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-drafts-'))
const { readComposerDraft, writeComposerDraft, clearComposerDraft, composerDraftFile, purgeComposerDrafts } = await import('./composer-drafts')
const a = { surface: 'chat' as const, conversationId: 'conversation-a', hostId: null, incognito: false }
const b = { ...a, conversationId: 'conversation-b' }
const initial = await readComposerDraft(a)
assert.equal(initial.revision, 0)
assert.equal(initial.text, '')
const first = await writeComposerDraft({ scope: a, expectedRevision: 0, text: 'Synthetic draft A', attachments: [] })
assert.equal(first.saved, true)
await writeComposerDraft({ scope: b, expectedRevision: 0, text: 'Synthetic draft B', attachments: [] })
assert.equal((await readComposerDraft(a)).text, 'Synthetic draft A')
assert.equal((await readComposerDraft(b)).text, 'Synthetic draft B')
const filename = composerDraftFile(a)
assert.match(readFileSync(filename, 'utf8'), /safe-json:v1:/)
assert.ok(!readFileSync(filename, 'utf8').includes('Synthetic draft A'))
assert.equal(statSync(filename).mode & 0o777, 0o600)
const stale = await writeComposerDraft({ scope: a, expectedRevision: 0, text: 'stale data', attachments: [] })
assert.equal(stale.saved, false)
assert.equal((await readComposerDraft(a)).text, 'Synthetic draft A')
const results = await Promise.all([
  writeComposerDraft({ scope: a, expectedRevision: 1, text: 'winner A', attachments: [] }),
  writeComposerDraft({ scope: a, expectedRevision: 1, text: 'winner B', attachments: [] }),
])
assert.equal(results.filter((result) => result.saved).length, 1, 'CAS validates inside serialized transaction')
const beforeClear = await readComposerDraft(a)
assert.equal((await clearComposerDraft({ scope: a, expectedRevision: 1 })).saved, false)
assert.equal((await readComposerDraft(a)).text, beforeClear.text)
assert.equal((await clearComposerDraft({ scope: a, expectedRevision: beforeClear.revision })).saved, true)
assert.equal((await readComposerDraft(a)).text, '')
const remote = { ...b, hostId: 'other-host' }
assert.equal((await readComposerDraft(remote)).text, '', 'Same thread ID on another host never sees local draft')
const attachments = [{ id: 'one', name: 'unreadable.png', kind: 'image' as const, mime: 'image/png', bytes: 10, status: 'needs-reselection' as const, error: 'Select again after restart.' }]
await writeComposerDraft({ scope: b, expectedRevision: 1, text: 'Reference', attachments })
assert.deepEqual((await readComposerDraft(b)).attachments, attachments)
const restarted = spawnSync(process.execPath, ['-e', `
  const assert = require('node:assert/strict');
  const { createJiti } = require('jiti');
  createJiti(process.cwd() + '/draft-restart-probe.cjs').import('./electron/store/composer-drafts').then(async ({readComposerDraft}) => {
    const scope = {surface:'chat',conversationId:'conversation-a',hostId:null,incognito:false};
    assert.equal((await readComposerDraft(scope)).text, '');
    const second = await readComposerDraft({...scope, conversationId:'conversation-b'});
    assert.equal(second.text, 'Reference');
    assert.equal(second.attachments[0].name, 'unreadable.png');
    console.log('Fresh process recovered both protected drafts and file reference.');
  }).catch(error => { console.error(error); process.exitCode = 1; });
`], { cwd: process.cwd(), env: process.env, encoding: 'utf8', timeout: 15_000 })
assert.equal(restarted.status, 0, restarted.stderr)
assert.match(restarted.stdout, /Fresh process recovered/)
await assert.rejects(() => writeComposerDraft({ scope: b, expectedRevision: 2, text: '', attachments: [{ ...attachments[0], dataUrl: 'data:image/png;base64,secret' } as any] }), /Unsupported|attachment/)
await assert.rejects(() => writeComposerDraft({ scope: b, expectedRevision: 2, text: '', attachments: [{ ...attachments[0], sourcePath: '/private/file' } as any] }), /Unsupported|attachment/)
const filesBefore = readdirSync(path.dirname(filename)).sort()
const privateScope = { ...a, conversationId: 'private-chat', incognito: true }
await assert.rejects(() => writeComposerDraft({ scope: privateScope, expectedRevision: 0, text: 'incognito contents', attachments: [] }), /Incognito/)
assert.deepEqual(readdirSync(path.dirname(filename)).sort(), filesBefore, 'Incognito does not create file or key artifacts')
// Real protected-state corruption must fail closed and retain exact recoverable bytes.
const original = readFileSync(filename, 'utf8')
writeFileSync(filename, '{ malformed')
await assert.rejects(() => readComposerDraft(a), /read|recover|unavailable|damaged/i)
await assert.rejects(() => writeComposerDraft({ scope: a, expectedRevision: 0, text: 'overwrite', attachments: [] }), /read|recover|unavailable|damaged/i)
assert.equal(readFileSync(filename, 'utf8'), '{ malformed')
writeFileSync(filename, original)
assert.equal((await readComposerDraft(a)).text, '')
let ownerChecks = 0
// Use the actual current revision for the encryption-time ownership test.
const guarded = await readComposerDraft(a)
ownerChecks = 0
await assert.rejects(() => writeComposerDraft({ scope: a, expectedRevision: guarded.revision, text: 'queued private data', attachments: [] }, () => {
  ownerChecks++
  if (ownerChecks > 1) throw new Error('Incognito transition rejected queued normal save')
}), /Incognito transition/)
assert.equal((await readComposerDraft(a)).text, '', 'Current owner is checked again immediately before encryption')
const queued = writeComposerDraft({ scope: a, expectedRevision: guarded.revision, text: 'do not resurrect', attachments: [] })
const purged = purgeComposerDrafts()
await assert.rejects(() => queued, /cleared/)
await purged
assert.equal(readdirSync(path.join(process.env.VONTRA_HOME!, 'state')).includes('composer-drafts'), false, 'Explicit forget clears draft envelopes and recovery copies')
const freshIncognitoProfile = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-incognito-drafts-'))
process.env.VONTRA_HOME = freshIncognitoProfile
await assert.rejects(() => readComposerDraft(privateScope), /Incognito/)
await assert.rejects(() => writeComposerDraft({ scope: privateScope, expectedRevision: 0, text: 'memory only', attachments: [] }), /Incognito/)
assert.deepEqual(readdirSync(freshIncognitoProfile), [], 'Fresh incognito draft API never creates state directory or profile key')
const symlinkProfile = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-draft-link-'))
const unrelated = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-draft-unrelated-'))
process.env.VONTRA_HOME = symlinkProfile
mkdirSync(path.join(symlinkProfile, 'state'))
const sentinel = path.join(unrelated, path.basename(composerDraftFile(a)))
writeFileSync(sentinel, 'UNRELATED-DATA-MUST-REMAIN')
symlinkSync(unrelated, path.join(symlinkProfile, 'state', 'composer-drafts'))
await assert.rejects(() => readComposerDraft(a), /unavailable/)
await assert.rejects(() => writeComposerDraft({ scope: a, expectedRevision: 0, text: 'cannot escape profile', attachments: [] }), /unavailable/)
await purgeComposerDrafts()
assert.equal(existsSync(path.join(symlinkProfile, 'state', 'composer-drafts')), false)
assert.equal(readFileSync(sentinel, 'utf8'), 'UNRELATED-DATA-MUST-REMAIN', 'Purge removes hostile directory links without following them')
console.log('Protected composer drafts, two chats, host identity, atomic CAS, clear guards, references, incognito and corrupt-data recovery passed')
