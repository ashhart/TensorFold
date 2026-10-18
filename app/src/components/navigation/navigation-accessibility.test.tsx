import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import clsx from 'clsx'
import { installFixtureDom } from './accessibility-fixture'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  const output = ts.transpileModule(readFileSync(path.join(process.cwd(), file), 'utf8'), { compilerOptions }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed production import: ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
function harness() {
  const values: any[] = [], effects: (() => void)[] = [], deps: any[][] = [], cleanups: any[] = []
  let cursor = 0
  const hooks = {
    useState(initial: any) { const index = cursor++; if (!(index in values)) values[index] = typeof initial === 'function' ? initial() : initial; return [values[index], (next: any) => { values[index] = typeof next === 'function' ? next(values[index]) : next }] },
    useRef(initial: any) { const index = cursor++; return values[index] ??= { current: initial } },
    useId() { return `fixture-id-${cursor++}` },
    useMemo(build: any) { cursor++; return build() },
    useCallback(callback: any) { cursor++; return callback },
    useEffect(effect: any, next: any[]) { const index = cursor++; if (!deps[index] || !next || next.some((v, i) => v !== deps[index][i])) { effects.push(() => { cleanups[index]?.(); cleanups[index] = effect() }); deps[index] = next } },
    useLayoutEffect(effect: any, next: any[]) { hooks.useEffect(effect, next) },
  }
  return { hooks, render<T>(build: () => T): T { cursor = 0; return build() }, flush() { while (effects.length) effects.shift()!() }, cleanup() { cleanups.forEach((cleanup) => cleanup?.()) } }
}
function entries(node: any, ancestors: any[] = []): { node: any; ancestors: any[] }[] {
  if (!node || typeof node !== 'object') return []
  return [{ node, ancestors }, ...React.Children.toArray(node.props?.children).flatMap((child) => entries(child, [...ancestors, node]))]
}
const find = (tree: any, predicate: (node: any) => boolean) => entries(tree).find(({ node }) => predicate(node))?.node
const text = (tree: any): string => typeof tree === 'string' || typeof tree === 'number' ? String(tree) : React.Children.toArray(tree?.props?.children).map(text).filter(Boolean).join(' ').replace(/\s+/g, ' ').trim()
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve() }
const dom = installFixtureDom()
const { listeners } = dom
const failures: string[] = []
async function check(name: string, run: () => void | Promise<void>) { try { await run(); console.log(`PASS ${name}`) } catch (error) { failures.push(`${name}: ${(error as Error).message}`); console.error(`FAIL ${failures.at(-1)}`) } }

const paletteHooks = harness()
const commandCalls: string[] = []
let importCalls = 0
const state: any = {
  threads: [{ id: 'branch-fixture', title: 'Synthetic discussion', kind: 'chat' }], rooms: [], projects: [], settings: { providers: [], approvalMode: 'ask' }, view: { name: 'home' },
  newTask: () => commandCalls.push('task'), newChat() {}, openNewProject() {}, setView() {}, cycleApprovalMode() {}, toggleSidebar() {}, openThread: (id: string) => commandCalls.push(id), saveSettings() {}, applyThreadUpdate() {}, setNotice() {},
}
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const { useAccessibleDialog } = load('src/components/navigation/useAccessibleDialog.ts', { react: paletteHooks.hooks })
const paletteImports: Record<string, any> = {
  react: paletteHooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node, flushSync: (action: () => void) => action() }, clsx,
  '../lib/store': { useStore }, '../lib/icons': { Icon: () => null }, '../lib/thread-row': { threadPlace: () => 'Synthetic project' },
  '../lib/ipc': { api: { listImportableSessions: async () => [{ source: 'codex', sessionId: 'synthetic-session', title: 'Synthetic import', messageCount: 3 }], importExternalSession: async () => { importCalls++; return { id: 'imported-fixture' } } } },
  './navigation/useAccessibleDialog': { useAccessibleDialog },
  './search/search-events': { requestConversationSearch: (request: any) => { assert.deepEqual(request, { scope: 'all' }); assert.equal(paletteHooks.render(CommandPalette), null, 'The palette closes before message search is requested'); commandCalls.push('search-content') } },
}
const { CommandPalette } = load('src/components/CommandPalette.tsx', paletteImports)
paletteHooks.render(CommandPalette); paletteHooks.flush()
listeners.find((entry) => entry.name === 'vontra:open-palette')!.callback()
let palette = paletteHooks.render(CommandPalette); dom.mount(palette); paletteHooks.flush(); await settle(); palette = paletteHooks.render(CommandPalette)
await check('Command palette exposes a labelled modal and query', () => {
  const dialog = find(palette, (node) => node.props.role === 'dialog')
  assert.ok(dialog, 'Opening the actual palette must expose role=dialog')
  assert.equal(dialog.props['aria-modal'], true)
  assert.ok(find(palette, (node) => node.props.id === dialog.props['aria-labelledby']))
  const input = find(palette, (node) => node.type === 'input')
  assert.equal(input.props.role, 'combobox')
  assert.ok(input.props['aria-label'] || find(palette, (node) => node.type === 'label' && node.props.htmlFor === input.props.id), 'Query has a programmatic label')
  assert.ok(find(palette, (node) => node.props.id === input.props['aria-controls'] && node.props.role === 'listbox'))
  assert.ok(find(palette, (node) => node.props.id === input.props['aria-activedescendant'] && node.props.role === 'option'))
})
await check('Tab cannot leave an open command palette for background controls', () => {
  const input = find(palette, (node) => node.type === 'input')
  let prevented = false
  const key = { key: 'Tab', preventDefault() { prevented = true }, stopPropagation() {}, stopImmediatePropagation() {} }
  listeners.filter((entry) => entry.name === 'keydown' && entry.capture).forEach((entry) => entry.callback(key))
  if (!prevented) input.props.onKeyDown(key)
  assert.equal(prevented, true, 'Tab must be contained before browser focus can reach the background')
})
await check('Command selection works through a touch or assistive click', () => {
  const row = find(palette, (node) => text(node) === 'New task ⌘N' && (node.type === 'button' || node.props.role === 'option'))
  assert.ok(row, 'The original New task action remains available')
  assert.equal(typeof row.props.onClick, 'function', 'A result must execute a click, not depend on mouseDown')
  row.props.onClick(); assert.deepEqual(commandCalls, ['task'])
})
await check('External session import remains a working command', async () => {
  const row = find(palette, (node) => node.props.role === 'option' && text(node).includes('Import: Synthetic import')) ?? find(palette, (node) => node.type === 'button' && text(node).includes('Import: Synthetic import'))
  assert.ok(row); assert.equal(typeof row.props.onClick, 'function'); row.props.onClick(); await settle()
  assert.equal(importCalls, 1); assert.ok(commandCalls.includes('imported-fixture'))
})
await check('Message search command closes the palette before requesting all saved conversations', () => {
  const row = find(palette, (node) => node.props.role === 'option' && text(node).includes('Search message contents'))
  assert.ok(row); row.props.onClick(); assert.ok(commandCalls.includes('search-content'))
})
await check('Arrow keys announce the active command and an empty query has no selected result', () => {
  const input = find(palette, (node) => node.type === 'input')
  input.props.onKeyDown({ key: 'ArrowDown', preventDefault() {} })
  let next = paletteHooks.render(CommandPalette)
  const moved = find(next, (node) => node.type === 'input')
  // Selection commands close the palette above, so reopen it through its real event.
  if (!moved) {
    // Close-state cleanup releases the modal barrier before its opener runs.
    paletteHooks.flush()
    dom.body.children = [dom.root]
    listeners.find((entry) => entry.name === 'vontra:open-palette')!.callback()
    next = paletteHooks.render(CommandPalette); dom.mount(next); paletteHooks.flush()
  }
  const activeInput = find(next, (node) => node.type === 'input')
  activeInput.props.onKeyDown({ key: 'ArrowDown', preventDefault() {} })
  next = paletteHooks.render(CommandPalette)
  const selectedInput = find(next, (node) => node.type === 'input')
  assert.ok(find(next, (node) => node.props.id === selectedInput.props['aria-activedescendant'] && node.props['aria-selected'] === true))
  selectedInput.props.onChange({ target: { value: 'no-matching-synthetic-command' } })
  next = paletteHooks.render(CommandPalette)
  assert.equal(find(next, (node) => node.type === 'input').props['aria-activedescendant'], undefined)
  assert.match(text(find(next, (node) => node.props.role === 'status')), /No commands found/)
})
paletteHooks.cleanup()
dom.body.children = [dom.root]

const dialogHooks = harness()
const hook = load('src/components/navigation/useAccessibleDialog.ts', { react: dialogHooks.hooks }).useAccessibleDialog
const overlay = dom.element('div'), dialog = dom.element('div'), input = dom.element('input'), closeButton = dom.element('button'), hiddenButton = dom.element('button')
hiddenButton.hidden = true
dialog.append(input, hiddenButton, closeButton); overlay.append(dialog); dom.body.append(overlay)
const alreadyInert = dom.element('div'); alreadyInert.inert = true; dom.body.append(alreadyInert)
let closed = 0
const options: any = { open: true, onClose: () => closed++, dialogRef: { current: dialog }, initialFocusRef: { current: input } }
dom.trigger.focus()
dialogHooks.render(() => hook(options)); dialogHooks.flush()
await check('Body portal makes background inert and takes initial focus', () => {
  assert.equal(dom.root.inert, true); assert.equal(overlay.inert, false); assert.equal(dom.doc.activeElement, input)
})
await check('Tab and Shift Tab remain inside enabled visible dialog controls', () => {
  assert.equal(dom.key('Tab').prevented, true); assert.equal(dom.doc.activeElement, closeButton)
  dom.key('Tab'); assert.equal(dom.doc.activeElement, input)
  dom.key('Tab', { shiftKey: true }); assert.equal(dom.doc.activeElement, closeButton)
})
await check('Background navigation shortcuts are blocked while native copy and editing remain usable', () => {
  for (const key of ['n', 'b', ',', '[', ']', 'k', 'f']) {
    const event = dom.key(key, { metaKey: true }); assert.equal(event.prevented, true); assert.equal(event.stopped, true)
  }
  for (const key of ['a', 'c', 'v', 'x', 'z', 'ArrowLeft', 'Backspace']) {
    const event = dom.key(key, { ctrlKey: true }); assert.equal(event.prevented, false); assert.equal(event.stopped, false)
  }
})
await check('Programmatic background focus returns to the dialog', () => {
  dom.trigger.focus()
  listeners.filter((entry) => entry.name === 'focusin' && entry.capture).forEach((entry) => entry.callback({ target: dom.trigger }))
  assert.equal(dom.doc.activeElement, input)
})
await check('Escape closes once and restores the trigger and prior inert values', () => {
  const escape = dom.key('Escape'); assert.equal(escape.prevented, true); assert.equal(closed, 1)
  dialogHooks.cleanup(); assert.equal(dom.doc.activeElement, dom.trigger)
  assert.equal(dom.root.inert, false); assert.equal(alreadyInert.inert, true)
  assert.equal(listeners.some((entry) => entry.capture), false)
})

await check('Nested dialogs isolate the top dialog and restore the earlier dialog before the application', () => {
  const nestedHooks = harness()
  const nestedHook = load('src/components/navigation/useAccessibleDialog.ts', { react: nestedHooks.hooks }).useAccessibleDialog
  const firstOverlay = dom.element('div'), firstDialog = dom.element('div'), firstInput = dom.element('input')
  const secondOverlay = dom.element('div'), secondDialog = dom.element('div'), secondInput = dom.element('input')
  firstDialog.append(firstInput); firstOverlay.append(firstDialog); secondDialog.append(secondInput); secondOverlay.append(secondDialog)
  dom.body.append(firstOverlay, secondOverlay)
  const first: any = { open: true, onClose() {}, dialogRef: { current: firstDialog } }
  const second: any = { open: false, onClose() {}, dialogRef: { current: secondDialog } }
  const render = () => nestedHooks.render(() => { nestedHook(first); nestedHook(second) })
  dom.trigger.focus(); render(); nestedHooks.flush()
  assert.equal(dom.doc.activeElement, firstInput)
  second.open = true; render(); nestedHooks.flush()
  assert.equal(firstOverlay.inert, true); assert.equal(secondOverlay.inert, false); assert.equal(dom.doc.activeElement, secondInput)
  dom.key('Tab'); assert.equal(dom.doc.activeElement, secondInput)
  second.open = false; render(); nestedHooks.flush()
  assert.equal(dom.doc.activeElement, firstInput); assert.equal(dom.root.inert, true)
  first.open = false; render(); nestedHooks.flush()
  assert.equal(dom.doc.activeElement, dom.trigger); assert.equal(dom.root.inert, false)
  nestedHooks.cleanup()
})

const turnHooks = harness()
const checkpoints = Array.from({ length: 10 }, (_, index) => ({ turn: index + 1, messageCount: index * 2, fileRef: null, fileSha: null, fingerprint: `fixture-${index}`, createdAt: index + 1 }))
const mutations: string[] = [], confirmations: string[] = [], branchOpens: string[] = []
let confirmResult = true, revertResult: any = { ok: true }, forkResult: any = { ok: true }, checkpointResult: any = checkpoints, changed = 0
const originalConfirm = Object.getOwnPropertyDescriptor(globalThis, 'confirm')
Object.defineProperty(globalThis, 'confirm', { configurable: true, value: (question: string) => { confirmations.push(question); return confirmResult } })
const turnState: any = { projects: [], threads: [], status: {}, approvals: {}, openThread: (id: string) => branchOpens.push(id), forkThreadFromTurn: async (_id: string, turn: number, options: any) => { mutations.push(`fork:${turn}:${options?.worktree ?? false}`); return forkResult } }
const { TurnsSection } = load('src/components/thread-panel/TurnsSection.tsx', {
  react: turnHooks.hooks, 'react/jsx-runtime': jsx,
  '../../lib/store/types': { BUSY_RUN: new Set(['thinking', 'streaming', 'tool', 'awaiting']) },
  '../../lib/store': { useStore: Object.assign((select: any) => select(turnState), { getState: () => turnState }) },
  '../../lib/ipc': { api: { checkpointTurns: async () => checkpointResult, checkpointRevertTurn: async ({ turn }: any) => { mutations.push(`revert:${turn}`); return revertResult } } }, '../../lib/icons': { Icon: () => null },
  '../../lib/thread-workspace': { effectiveThreadCwd: (thread: any) => thread.cwd ?? '', effectiveThreadWorkspace: (thread: any) => thread.workspace ?? null }, './primitives': { PanelSection: 'section' },
})
const props: any = { thread: { id: 'synthetic-thread', title: 'Synthetic chat', kind: 'chat', updatedAt: 1 }, busy: false, onChanged() { changed++ } }
turnState.threads = [props.thread]
turnHooks.render(() => TurnsSection(props)); turnHooks.flush(); await settle()
const turnsTree = turnHooks.render(() => TurnsSection(props))
await check('Fork and Revert are visible and focusable without hover', () => {
  const actions = entries(turnsTree).filter(({ node }) => node.type === 'button' && /^(Fork|Revert)/.test(text(node)))
  assert.ok(actions.length >= 2)
  for (const { node, ancestors } of actions) {
    assert.equal(ancestors.some((ancestor) => /(^|\s)hidden(\s|$)/.test(ancestor.props.className ?? '')), false, 'Actual action ancestor must not use display:none until hover')
    assert.ok(node.props['aria-label']?.match(/turn \d+/), 'Repeated actions identify their turn')
    assert.match(node.props.className, /focus-visible:/, 'The action has a visible keyboard focus indicator')
  }
})
await check('Chat Revert explanation never promises file restoration', () => {
  const revert = find(turnsTree, (node) => node.type === 'button' && text(node) === 'Revert')
  assert.ok(revert); assert.match(revert.props.title, /Restores the conversation/); assert.match(revert.props.title, /No workspace files are changed/)
})
await check('All saved turns are reachable without truncating the chooser', () => {
  assert.match(text(turnsTree), /Turn 1\b/, 'An early checkpoint must be available, not silently sliced away')
})
const renderTurns = () => turnHooks.render(() => TurnsSection(props))
const turnAction = (label: string, tree = renderTurns()) => find(tree, (node) => node.type === 'button' && node.props['aria-label'] === label)
await check('Revert cancellation performs no mutation and names the exact scope', async () => {
  confirmResult = false
  await turnAction('Revert to before turn 1').props.onClick()
  assert.deepEqual(mutations, []); assert.match(confirmations.at(-1)!, /before turn 1/); assert.match(confirmations.at(-1)!, /No workspace files are changed/)
  confirmResult = true
})
await check('Normal revert calls the existing API once and announces completion', async () => {
  await turnAction('Revert to before turn 1').props.onClick()
  assert.deepEqual(mutations, ['revert:1']); assert.equal(changed, 1)
  assert.match(text(renderTurns()), /Conversation restored to before turn 1/)
})
await check('Pending fork rejects duplicate activations before a React rerender', async () => {
  let finish!: (result: any) => void
  forkResult = new Promise((resolve) => { finish = resolve })
  const action = turnAction('Fork conversation before turn 2')
  const first = action.props.onClick(), duplicate = action.props.onClick()
  assert.equal(mutations.filter((call) => call === 'fork:2:false').length, 1)
  finish({ ok: true }); await first; await duplicate
  forkResult = { ok: true }
})
await check('A newly waiting approval or busy run rejects an old action callback', async () => {
  const action = turnAction('Revert to before turn 3'), count = mutations.length
  turnState.approvals[props.thread.id] = [{ callId: 'synthetic-call', itemId: 'synthetic-item' }]
  await action.props.onClick(); assert.equal(mutations.length, count)
  turnState.approvals = {}; turnState.status[props.thread.id] = 'streaming'
  await action.props.onClick(); assert.equal(mutations.length, count)
  turnState.status = {}
})
await check('Busy UI keeps every turn action visible and disabled', () => {
  props.busy = true
  const tree = renderTurns()
  for (const { node } of entries(tree).filter(({ node }) => node.type === 'button' && /^(Fork|Revert)/.test(text(node)))) assert.equal(node.props.disabled, true)
  props.busy = false; renderTurns()
})
await check('Host refusal and partial restoration warning remain visible', async () => {
  revertResult = { ok: false, error: 'Synthetic checkpoint refusal.' }
  await turnAction('Revert to before turn 4').props.onClick()
  assert.match(text(renderTurns()), /Synthetic checkpoint refusal/)
  revertResult = { ok: true, warning: 'Synthetic file snapshot unavailable. Conversation restored only.' }
  await turnAction('Revert to before turn 4').props.onClick()
  assert.match(text(renderTurns()), /Conversation restored only/)
  revertResult = { ok: true }
})
await check('Branch chooser opens the original thread without modifying either prompt', () => {
  const source = { ...props.thread, id: 'synthetic-source', title: 'Original prompt branch', timeline: [{ text: 'Original question' }] }
  const sibling = { ...props.thread, id: 'synthetic-sibling', parentId: source.id, title: 'Alternative answer branch', timeline: [{ text: 'Different question' }] }
  props.thread = { ...props.thread, parentId: source.id, timeline: [{ text: 'Current question' }] }
  turnState.threads = [props.thread, source, sibling]
  const before = JSON.stringify(turnState.threads), tree = renderTurns()
  assert.ok(find(tree, (node) => node.props['aria-current'] === 'page' && text(node).includes('Synthetic chat')))
  find(tree, (node) => node.type === 'button' && text(node).includes('Original prompt branch')).props.onClick()
  assert.deepEqual(branchOpens, [source.id]); assert.equal(JSON.stringify(turnState.threads), before)
})
await check('A changed thread revision invalidates old turn controls while refreshing snapshots', async () => {
  const oldAction = turnAction('Revert to before turn 1'), count = mutations.length
  props.thread = { ...props.thread, updatedAt: 2 }; turnState.threads[0] = props.thread
  await oldAction.props.onClick(); assert.equal(mutations.length, count)
  renderTurns(); turnHooks.flush()
  assert.equal(turnAction('Revert to before turn 1').props.disabled, true)
  await settle(); assert.equal(turnAction('Revert to before turn 1').props.disabled, false)
})
await check('Empty and failed checkpoint loads have explicit states after navigation', async () => {
  checkpointResult = []
  props.thread = { ...props.thread, id: 'synthetic-empty', updatedAt: 1 }; turnState.threads = [props.thread]
  renderTurns(); turnHooks.flush(); await settle()
  assert.match(text(renderTurns()), /No saved turns yet/)
  checkpointResult = Promise.reject(new Error('Synthetic turn load failure.'))
  props.thread = { ...props.thread, updatedAt: 2 }; turnState.threads = [props.thread]
  renderTurns(); turnHooks.flush(); await settle()
  assert.match(text(renderTurns()), /Synthetic turn load failure/)
})
turnHooks.cleanup()
if (originalConfirm) Object.defineProperty(globalThis, 'confirm', originalConfirm); else Reflect.deleteProperty(globalThis, 'confirm')
dom.restore()
if (failures.length) throw new Error(`${failures.length} accessibility regressions:\n${failures.join('\n')}`)
console.log('Actual navigation components pass synthetic keyboard, accessible result, touch activation and checkpoint discovery checks.')
