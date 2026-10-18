import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { liveMessagingDraftKey, telegramMessagingDraftKey } from './useMessagingDraftState'

const slots: any[] = []
let cursor = 0
const react = {
  useState: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
  useRef: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
  useEffect: (work: () => void) => work(),
}
const output = ts.transpileModule(readFileSync(path.join(__dirname, 'useMessagingDraftState.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', output)((name: string) => { assert.equal(name, 'react'); return react }, loaded, loaded.exports)
const initial = { id: 'source_a', label: 'Example source', token: '••••••••', channels: 'C123', enabled: true, channelPrompt: '', credentialVersion: 1 }
let draft = { ...initial }, persisted = { ...initial }
const render = () => { cursor = 0; return loaded.exports.useMessagingDraftState(draft.enabled, async () => {}, { draftKey: liveMessagingDraftKey(draft), persistedKey: liveMessagingDraftKey(persisted), enabled: persisted.enabled }) }
assert.equal(render().dirty, false)
render().edited(); draft = { ...draft, channels: 'C456', enabled: false }
assert.equal(render().dirty, true)
assert.equal(render().persistedEnabled, true, 'local disable does not claim the saved host connection stopped')
const withStatus = { ...persisted, lastSyncAt: 100, lastError: 'Example failure', updatedAt: 200 }
assert.equal(liveMessagingDraftKey(withStatus), liveMessagingDraftKey(persisted), 'host sync/status fields do not count as config saves')
persisted = withStatus
assert.equal(render().dirty, true, 'unrelated host metadata cannot clear a source draft')
persisted = { ...draft }
assert.equal(render().dirty, false, 'global Save reconciles without calling card.save or remounting')
assert.equal(render().persistedEnabled, false)
render().edited(); draft = { ...draft, label: 'Unfinished new label' }
assert.equal(render().dirty, true)
draft = { ...persisted }
assert.equal(render().dirty, false, 'global Discard clears local helper state from source config equality')
render().edited(); draft = { ...draft, token: 'new-synthetic-token', credentialVersion: 1 }
persisted = { ...draft, token: '••••••••', credentialVersion: 2 }
assert.equal(render().dirty, true, 'a newly acknowledged host secret does not erase a still-different local token draft')
draft = { ...persisted }
assert.equal(render().dirty, false, 'redacted host/draft reconciliation clears a saved token edit')
render().edited(); draft = { ...draft, channels: 'C789' }
persisted = { ...persisted, label: 'Earlier save acknowledged' }
assert.equal(render().dirty, true, 'a save acknowledging an older source configuration leaves newer edits pending')
assert.notEqual(liveMessagingDraftKey(initial), liveMessagingDraftKey({ ...initial, id: 'source_b' }), 'same-looking accounts retain separate identities')
const telegram = { enabled: true, botToken: '••••••••', allowedChatIds: '12345', allowedUserIds: '12345', polling: false, secretRotations: { botToken: 1 } }
assert.equal(telegramMessagingDraftKey(telegram), telegramMessagingDraftKey({ ...telegram, lastPollAt: 100, lastUpdateId: 50, lastError: 'status only', remote: { enabled: true } } as typeof telegram), 'runtime polling and separate remote-scope updates cannot erase bot setup edits')
assert.notEqual(telegramMessagingDraftKey(telegram), telegramMessagingDraftKey({ ...telegram, allowedUserIds: '54321' }))
console.log('Global Settings Save/Discard, redacted token reconciliation, source-specific draft retention and ignored host metadata regressions passed')
