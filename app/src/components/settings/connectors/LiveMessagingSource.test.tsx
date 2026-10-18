import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import * as connectionState from './messaging-connection-state'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = (_props: any) => null, Field = (_props: any) => null, Input = (_props: any) => null
const persistenceCompiled = ts.transpileModule(readFileSync(path.join(__dirname, 'useMessagingDraftState.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const compiled = ts.transpileModule(readFileSync(path.join(__dirname, 'LiveMessagingSource.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
function harness(id: string) {
  const slots: any[] = []
  const effects: Array<() => void> = []
  const cleanups: Array<() => void> = []
  let cursor = 0, initialized = false
  let value: any = { id, label: id, token: 'fixture-secret', channels: 'C123', enabled: false, lastSyncAt: null }
  let waitForSave: Promise<void> | null = null
  let failSave = false
  let persistedValue: any | undefined
  const calls: string[] = []
  const imports: Record<string, unknown> = {
    react: {
      useState: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
      useRef: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
      useEffect: (work: () => void) => { if (!initialized) effects.push(work) },
    },
    'react/jsx-runtime': jsx,
    '../../../lib/brand': { BrandIcon: () => null },
    '../../ui': { Button, Field, Input },
    './messaging-connection-state': connectionState,
  }
  const persistence = { exports: {} as any }
  new Function('require', 'module', 'exports', persistenceCompiled)((name: string) => { assert.equal(name, 'react'); return imports.react }, persistence, persistence.exports)
  imports['./useMessagingDraftState'] = persistence.exports
  const loaded = { exports: {} as any }
  new Function('require', 'module', 'exports', compiled)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, `Reviewed import ${name}`); return imports[name] }, loaded, loaded.exports)
  const render = () => {
    cursor = 0
    const node = loaded.exports.LiveMessagingSource({ brand: 'slack', value, persistedValue,
      edit: (patch: any) => { value = { ...value, ...patch } },
      save: async (patch: any) => { calls.push('save'); if (waitForSave) await waitForSave; if (failSave) throw new Error('Fixture save failure'); value = { ...value, ...patch }; if (persistedValue !== undefined) persistedValue = structuredClone(value) },
      verify: async () => { calls.push('verify'); return { ok: true, identity: id } },
      listChannels: async () => { calls.push('channels'); return [{ id: 'C123', name: 'first' }, { id: 'C456', name: 'second' }] },
      sync: async () => { calls.push('sync'); return 'Read selected channels only' },
      remove: async () => { calls.push('remove') },
    })
    if (!initialized) { for (const effect of effects) { const cleanup = effect() as unknown; if (typeof cleanup === 'function') cleanups.push(cleanup as () => void) } initialized = true }
    return node
  }
  return { render, calls, wait: (promise: Promise<void> | null) => { waitForSave = promise }, value: () => value, initialHost: (patch: any) => { value = { ...value, ...patch }; persistedValue = structuredClone(value) }, globalSave: () => { persistedValue = structuredClone(value) }, globalDiscard: () => { value = structuredClone(persistedValue) }, hostMetadata: (patch: any) => { persistedValue = { ...persistedValue, ...patch } }, failSave: (value: boolean) => { failSave = value }, close: () => cleanups.forEach((cleanup) => cleanup()) }
}
function find(node: any, match: (item: any) => boolean): any {
  if (!node || typeof node !== 'object') return undefined
  if (match(node)) return node
  for (const child of React.Children.toArray(node.props?.children)) { const result = find(child, match); if (result) return result }
}
const button = (node: any, text: string) => find(node, (item) => item.type === Button && item.props.children === text)
const a = harness('workspace-a'), b = harness('workspace-b')
try {
  let screen = a.render()
  assert.equal(button(screen, 'Connect selected channels').props.disabled, true)
  assert.equal(button(screen, 'Read selected channels').props.disabled, true)
  assert.equal(find(screen, (item) => item.type === Input && item.props['aria-label'] === 'Slack bot token').props.type, 'password')
  let releaseSave!: () => void
  a.wait(new Promise<void>((resolve) => { releaseSave = resolve }))
  const checking = button(screen, 'Save and verify token').props.onClick()
  await Promise.resolve()
  assert.deepEqual(a.calls, ['save'], 'actual button handler saves durably before host verification')
  assert.equal(button(a.render(), 'Verifying…').props.disabled, true)
  assert.equal(button(b.render(), 'Connect selected channels').props.disabled, true, 'second account remains unverified')
  releaseSave(); await checking; a.wait(null)
  screen = a.render()
  assert.equal(button(screen, 'Connect selected channels').props.disabled, false)
  assert.equal(button(b.render(), 'Connect selected channels').props.disabled, true)
  await button(screen, 'Choose channels').props.onClick()
  screen = a.render()
  const secondChannel = find(screen, (item) => item.type === 'input' && item.props.type === 'checkbox' && item.props.checked === false)
  secondChannel.props.onChange({ target: { checked: true } })
  assert.match(a.value().channels, /C456/)
  assert.equal(a.value().enabled, false)
  screen = a.render()
  await button(screen, 'Connect selected channels').props.onClick()
  assert.equal(a.value().enabled, true)
  screen = a.render()
  await button(screen, 'Read selected channels').props.onClick()
  assert.deepEqual(a.calls.slice(-2), ['save', 'sync'], 'actual read handler waits for settings persistence before sync')
  find(a.render(), (item) => item.type === Input && item.props['aria-label'] === 'Slack bot token').props.onChange({ target: { value: 'replacement-secret' } })
  screen = a.render()
  assert.equal(a.value().enabled, false)
  assert.equal(button(screen, 'Read selected channels').props.disabled, true, 'editing token invalidates the account before read')
  assert.equal(button(screen, 'Disconnect').props.disabled, false, 'editing a local disabled draft keeps saved active connection disconnect available')
  assert.ok(find(screen, (item) => item.type === 'p' && React.Children.toArray(item.props.children).includes('Unsaved changes · Saved connection remains enabled')))
  a.failSave(true)
  await button(screen, 'Disconnect').props.onClick()
  screen = a.render()
  assert.equal(button(screen, 'Disconnect').props.disabled, false, 'failed persistence cannot claim the saved connection stopped')
  a.failSave(false)
  await button(screen, 'Disconnect').props.onClick()
  screen = a.render()
  assert.equal(button(screen, 'Disconnect').props.disabled, true, 'successful host save is required before showing disconnected')
  const global = harness('global-settings-source')
  try {
    global.initialHost({ enabled: true })
    let view = global.render()
    find(view, (item) => item.type === Input && item.props['aria-label'] === 'Slack allowed channels').props.onChange({ target: { value: 'C456' } })
    view = global.render()
    assert.equal(button(view, 'Disconnect').props.disabled, false)
    global.hostMetadata({ lastSyncAt: 100, lastError: 'status-only change' })
    view = global.render()
    assert.ok(find(view, (item) => item.props?.role === 'status' && String(item.props.children).includes('previous saved settings')), 'status updates never clear the unsaved notice')
    global.globalSave()
    view = global.render()
    assert.equal(button(view, 'Disconnect').props.disabled, true, 'global Save of the disabled draft updates actual connection state')
    assert.equal(find(view, (item) => item.props?.role === 'status' && String(item.props.children).includes('previous saved settings')), undefined)
    find(view, (item) => item.type === Input && item.props['aria-label'] === 'Slack connection label').props.onChange({ target: { value: 'Unsaved label' } })
    view = global.render()
    assert.ok(find(view, (item) => item.props?.role === 'status' && String(item.props.children).includes('previous saved settings')))
    global.globalDiscard()
    view = global.render()
    assert.equal(find(view, (item) => item.props?.role === 'status' && String(item.props.children).includes('previous saved settings')), undefined, 'global Discard clears stale local card status without remounting')
  } finally { global.close() }
  assert.deepEqual(b.calls, [], 'first-account actions never invoke the second account')
  console.log('Actual messaging UI handlers: password input, save-before-verify/sync, independent accounts, picker, connect and token invalidation passed')
} finally { a.close(); b.close() }
