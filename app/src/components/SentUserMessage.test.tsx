import { reviewedLanguageImport } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { createEditedTaskBranch, prepareThreadMessageEdit, stageThreadMessageEdit } from '../../electron/agent/message-edit'
import type { Thread } from '../lib/types'

const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
let slots: any[] = [], refs: any[] = [], cursor = 0, refCursor = 0
let edited: any[] = [], prepared = 0, copied: string[] = [], updated: Thread[] = [], navigated: any[] = []
let mode: 'success' | 'reject' | 'pending' = 'success'
let release: (() => void) | null = null
let current!: Thread
const hooks = {
  useState(initial: any) { const slot = cursor++; if (!(slot in slots)) slots[slot] = typeof initial === 'function' ? initial() : initial; return [slots[slot], (value: any) => { slots[slot] = typeof value === 'function' ? value(slots[slot]) : value }] },
  useRef(initial: any) { const slot = refCursor++; return refs[slot] ??= { current: initial } },
}
const useStore = Object.assign((selector: any) => selector({ threads: [current] }), { getState: () => ({ applyThreadUpdate(thread: Thread) { updated.push(thread); if (thread.id === current.id) current = thread }, setView(view: any) { navigated.push(view) } }) })
const imports: Record<string, unknown> = {
  './composer/MessageQuotes': { MessageQuoteAction: () => null, SentMessageQuotes: () => null },
  react: hooks, 'react/jsx-runtime': jsx, '../lib/store': { useStore }, '../lib/icons': { Icon: () => null }, './timeline': { UserBubble: ({ text }: any) => React.createElement('div', null, text) },
  '../lib/ipc': { api: {
    writeClipboard: async (text: string) => { copied.push(text); return { ok: true } },
    prepareMessageEdit: async ({ itemId }: any) => { prepared++; return prepareThreadMessageEdit(current, itemId) },
    editMessage: async (input: any) => { edited.push(input); if (mode === 'reject') throw new Error('Synthetic durable edit rejected'); if (mode === 'pending') await new Promise<void>((resolve) => { release = resolve }); const patch = stageThreadMessageEdit(current, input); return current.kind === 'task' ? createEditedTaskBranch(current, patch, input.itemId) : { ...current, ...patch } },
  } },
}
const module = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'SentUserMessage.tsx'), 'utf8'), { compilerOptions }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed renderer import ${id}`); return imports[id] }, module, module.exports)
function fixture(): Thread { return { id: 'fixture-chat', title: 'Synthetic', kind: 'chat', cwd: '', providerId: null, model: null, status: 'idle', createdAt: 1, updatedAt: 1, apiMessages: [{ role: 'user', content: 'Exact\nmessage with  spaces' }, { role: 'assistant', content: 'Old answer' }], timeline: [{ id: 'u', kind: 'user', text: 'Exact\nmessage with  spaces' }, { id: 'a', kind: 'assistant', text: 'Old answer' }] } }
function reset() { current = fixture(); slots = []; refs = []; prepared = 0; edited = []; copied = []; updated = []; navigated = []; mode = 'success'; release = null }
function render(busy = false) { cursor = 0; refCursor = 0; return module.exports.SentUserMessage({ thread: current, item: current.timeline[0], busy }) }
function all(node: any, predicate: (value: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function button(node: any, label: string) { const result = all(node, (entry) => entry.type === 'button' && (entry.props.children === label || entry.props['aria-label'] === label))[0]; assert.ok(result, label); return result }
function textarea(node: any) { const result = all(node, (entry) => entry.type === 'textarea')[0]; assert.ok(result); return result }
async function settle() { for (let i = 0; i < 15; i++) await Promise.resolve() }
const originalWindow = globalThis.window
Object.defineProperty(globalThis, 'window', { configurable: true, value: { setTimeout: () => 1 } })
try {
  reset()
  let node = render()
  button(node, 'Copy user message').props.onClick(); await settle()
  assert.deepEqual(copied, ['Exact\nmessage with  spaces'])
  const beforeCancel = structuredClone(current)
  button(node, 'Edit user message').props.onClick(); button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  assert.equal(prepared, 1, 'duplicate opening is synchronously rejected')
  assert.match(renderToStaticMarkup(node), /Earlier revisions/)
  assert.match(renderToStaticMarkup(node), /external actions are not undone/)
  textarea(node).props.onChange({ target: { value: 'Cancelled edit' } }); node = render()
  button(node, 'Cancel').props.onClick(); node = render()
  assert.deepEqual(current, beforeCancel)
  assert.equal(edited.length, 0)
  button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  assert.equal(textarea(node).props.value, beforeCancel.timeline[0].text)
  textarea(node).props.onChange({ target: { value: 'Revised message' } }); node = render()
  mode = 'reject'; button(node, 'Save changes').props.onClick(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Synthetic durable edit rejected/)
  assert.equal(textarea(node).props.value, 'Revised message', 'failed save retains the editable text')
  assert.deepEqual(current, beforeCancel, 'failed save changes neither displayed model history nor durable acceptance')
  mode = 'success'; button(node, 'Save changes').props.onClick(); await settle(); node = render()
  assert.equal(current.timeline[0].text, 'Revised message')
  assert.equal(current.apiMessages[0].content, 'Revised message')
  assert.equal(current.apiMessages.length, 1)
  assert.equal(current.messageEditRevisions?.[0].timeline[1].text, 'Old answer')
  assert.equal(updated.length, 1)
  assert.equal(navigated.length, 0)
  // Active generation disables action in both rendered controls and direct event handlers.
  reset(); node = render(true)
  assert.equal(button(node, 'Edit user message').props.disabled, true)
  button(node, 'Edit user message').props.onClick(); await settle(); assert.equal(prepared, 0)
  node = render(); button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  textarea(node).props.onChange({ target: { value: 'Busy edit' } }); node = render(true)
  assert.equal(button(node, 'Save changes').props.disabled, true)
  button(node, 'Save changes').props.onClick(); await settle(); assert.equal(edited.length, 0)
  // Pending commit cannot be cancelled or submitted twice before rerender.
  reset(); node = render(); button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  textarea(node).props.onChange({ target: { value: 'Pending edit' } }); node = render()
  mode = 'pending'; button(node, 'Save changes').props.onClick(); button(node, 'Save changes').props.onClick(); button(node, 'Cancel').props.onClick(); await settle(); node = render()
  assert.equal(edited.length, 1)
  assert.equal(button(node, 'Cancel').props.disabled, true)
  assert.equal(button(node, 'Saving…').props.disabled, true)
  assert.equal(textarea(node).props.value, 'Pending edit')
  ;(release as (() => void) | null)?.(); await settle()
  // Task edit clearly creates a branch and navigates to it; original source stays intact.
  reset(); current.kind = 'task'; const source = structuredClone(current)
  node = render(); button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /edited branch[\s\S]*checkpoints/)
  assert.match(renderToStaticMarkup(node), /shares the current workspace files/)
  textarea(node).props.onChange({ target: { value: 'New task request' } }); node = render()
  button(node, 'Save as edited branch').props.onClick(); await settle()
  assert.deepEqual(current, source)
  assert.notEqual(updated[0].id, source.id)
  assert.deepEqual(navigated, [{ name: 'thread', id: updated[0].id }])
  // Visible remote limitation is actionable and makes no host mutation.
  reset(); current.orchestratorHostId = 'fixture-host'; node = render()
  button(node, 'Edit user message').props.onClick(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /owning host/); assert.equal(prepared, 0)
  console.log('Actual sent-message handlers: exact Copy, explicit Save/Cancel, failure-retained text, active and duplicate guards, pending commit, model/timeline consistency and task-branch navigation passed')
} finally { Object.defineProperty(globalThis, 'window', { configurable: true, value: originalWindow }) }
