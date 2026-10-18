import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import * as connectionState from './messaging-connection-state'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = (_props: any) => null, Field = (_props: any) => null, Input = (_props: any) => null, Toggle = (_props: any) => null, inert = () => null
const compile = (file: string) => ts.transpileModule(readFileSync(path.join(__dirname, file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const slots: any[] = [], effects: Array<() => void> = [], cleanups: Array<() => void> = []
let cursor = 0, initialized = false, failingSave = false
let draft: any = { botControl: { telegram: { enabled: true, botToken: 'fixture-bot-token', allowedChatIds: '12345', allowedUserIds: '12345', polling: true, lastUpdateId: null, lastPollAt: null, lastError: null } } }
let hostTelegram = structuredClone(draft.botControl.telegram)
const apiCalls: string[] = []
const react = {
  useState: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
  useRef: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
  useEffect: (work: () => void) => { if (!initialized) effects.push(work) },
}
const persistence = { exports: {} as any }
new Function('require', 'module', 'exports', compile('useMessagingDraftState.ts'))((name: string) => { assert.equal(name, 'react'); return react }, persistence, persistence.exports)
const imports: Record<string, unknown> = {
  react, 'react/jsx-runtime': jsx, '../../../lib/ipc': { api: {
    verifyTelegram: async () => { apiCalls.push('verify'); return { ok: true, username: 'fixture_bot' } },
    pollTelegram: async () => { apiCalls.push('poll'); assert.equal(hostTelegram.polling, false, 'manual checking never needs or enables background polling'); return { ok: true, processed: 1, skipped: 2 } },
  } }, '../../../lib/brand': { BrandIcon: inert }, '../../ui': { Button, Field, Input, Toggle }, '../shared': { Section: inert },
  './MessagingSetupGuide': { MessagingSetupGuide: inert }, './messaging-connection-state': connectionState, './TelegramRemoteSettingsPanel': { TelegramRemoteSettingsPanel: inert }, './useMessagingDraftState': persistence.exports,
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', compile('TelegramConnector.tsx'))((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports); return imports[name] }, loaded, loaded.exports)
const render = () => {
  cursor = 0
  const node = loaded.exports.TelegramConnector({ ctx: { draft, persistedSettings: { botControl: { telegram: hostTelegram } }, setDraft: (update: any) => { draft = typeof update === 'function' ? update(draft) : update }, save: async (next: any) => { if (failingSave) throw new Error('Fixture save failure'); draft = next; hostTelegram = structuredClone(next.botControl.telegram) } } })
  if (!initialized) { for (const effect of effects) { const cleanup: unknown = effect(); if (typeof cleanup === 'function') cleanups.push(cleanup as () => void) } initialized = true }
  return node
}
function find(node: any, match: (item: any) => boolean): any {
  if (!node || typeof node !== 'object') return undefined
  if (match(node)) return node
  for (const child of React.Children.toArray(node.props?.children)) { const result = find(child, match); if (result) return result }
}
const button = (node: any, text: string) => find(node, (item) => item.type === Button && item.props.children === text)
try {
  let node = render()
  assert.equal(apiCalls.length, 0, 'mount never verifies, polls or sends automatically')
  find(node, (item) => item.type === Input && item.props['aria-label'] === 'Telegram bot token').props.onChange({ target: { value: '' } })
  node = render()
  assert.equal(draft.botControl.telegram.enabled, false)
  assert.equal(hostTelegram.enabled, true, 'editing local settings does not itself stop the host')
  assert.equal(button(node, 'Disconnect').props.disabled, false, 'saved active connection remains disconnectable')
  assert.equal(button(node, 'Remove token').props.disabled, false, 'clearing an unsaved token does not hide the saved-token removal action')
  failingSave = true; await button(node, 'Disconnect').props.onClick(); node = render()
  assert.equal(hostTelegram.enabled, true)
  assert.equal(button(node, 'Disconnect').props.disabled, false)
  failingSave = false; await button(node, 'Remove token').props.onClick(); node = render()
  assert.equal(hostTelegram.enabled, false)
  assert.equal(hostTelegram.botToken, '')
  find(node, (item) => item.type === Input && item.props['aria-label'] === 'Telegram bot token').props.onChange({ target: { value: 'replacement-fixture-token' } })
  node = render(); await button(node, 'Save and verify token').props.onClick(); node = render()
  await button(node, 'Connect bot').props.onClick(); node = render()
  assert.equal(hostTelegram.enabled, true)
  assert.equal(hostTelegram.polling, false)
  await button(node, 'Check now').props.onClick()
  assert.deepEqual(apiCalls, ['verify', 'poll'])
  node = render()
  find(node, (item) => item.type === Input && item.props['aria-label'] === 'Telegram allowed sender IDs').props.onChange({ target: { value: '*' } })
  node = render()
  assert.equal(button(node, 'Connect bot').props.disabled, true, 'sender access cannot silently expand to everyone')
  assert.equal(button(node, 'Check now').props.disabled, true)
  draft.botControl.telegram = structuredClone(hostTelegram)
  node = render()
  assert.equal(button(node, 'Check now').props.disabled, false, 'global Discard restores the actual enabled bot access without stale pending state')
  assert.equal(find(node, (item) => item.props?.role === 'status' && String(item.props.children).includes('These are unsaved edits')), undefined)
  find(node, (item) => item.type === Input && item.props['aria-label'] === 'Telegram allowed sender IDs').props.onChange({ target: { value: '54321' } })
  node = render()
  hostTelegram = structuredClone(draft.botControl.telegram)
  node = render()
  assert.equal(button(node, 'Disconnect').props.disabled, true, 'global Save of bot setup reconciles saved disabled state')
  assert.equal(find(node, (item) => item.props?.role === 'status' && String(item.props.children).includes('These are unsaved edits')), undefined)
  console.log('Telegram actual setup handlers: no automatic calls, truthful unsaved/disconnect state, token removal, manual polling and explicit sender scope passed')
} finally { cleanups.forEach((cleanup) => cleanup()) }
