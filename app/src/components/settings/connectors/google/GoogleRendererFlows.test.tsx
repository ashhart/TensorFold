import { reviewedLanguageImport } from '../../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import * as time from './google-calendar-time'

const Button = (_props: any) => null, Field = (_props: any) => null, Input = (_props: any) => null, Toggle = (_props: any) => null
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
function harness(file: string, api: any, extra: Record<string, unknown> = {}) {
  const slots: any[] = []
  let cursor = 0
  let effects: Array<() => void> = []
  const cleanups: Array<() => void> = []
  const react = {
    useState: (initial: any) => { const index = cursor++; if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
    useRef: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
    useCallback: (fn: any) => { cursor++; return fn },
    useEffect: (work: () => any, deps: unknown[]) => { const index = cursor++; const previous = slots[index]; if (!previous || deps.some((value, i) => value !== previous[i])) { slots[index] = deps; effects.push(() => { const cleanup = work(); if (typeof cleanup === 'function') cleanups.push(cleanup) }) } },
  }
  const load = (filename: string): any => {
    const compiled = ts.transpileModule(readFileSync(path.join(__dirname, filename), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
    const imports: Record<string, unknown> = { react, 'react/jsx-runtime': jsx, '../../../../lib/ipc': { api }, '../../../ui': { Button, Field, Input, Toggle }, './google-calendar-time': time, ...extra }
    if (filename !== 'useGoogleAccountRequest.ts') imports['./useGoogleAccountRequest'] = load('useGoogleAccountRequest.ts')
    const loaded = { exports: {} as any }
    new Function('require', 'module', 'exports', compiled)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, `Reviewed renderer import ${name}`); return imports[name] }, loaded, loaded.exports)
    return loaded.exports
  }
  const loaded = load(file)
  return {
    render(name: string, props: any) { cursor = 0; effects = []; const node = loaded[name](props); for (const effect of effects) effect(); return node },
    close: () => { cleanups.splice(0).forEach((cleanup) => cleanup()) },
  }
}
function find(node: any, match: (item: any) => boolean): any {
  if (!node || typeof node !== 'object') return undefined
  if (match(node)) return node
  for (const child of React.Children.toArray(node.props?.children)) { const result = find(child, match); if (result) return result }
}
const visibleText = (node: any): string => React.Children.toArray(node).map((child: any) => child && typeof child === 'object' ? visibleText(child.props?.children) : String(child ?? '')).join('')
const button = (node: any, text: string) => find(node, (item) => item.type === Button && item.props.children === text)
const input = (node: any, label: string) => find(node, (item) => item.type === Input && item.props['aria-label'] === label)
const account = (id: string) => ({ id, email: `${id}@example.com`, revision: 1, capabilities: ['gmail.read', 'calendar.read'], status: 'connected', connectedAt: 0 })
// Polling and permission refresh must reconcile one setup/tools/reviews pane each.
// Duplicate sibling keys previously caused setup cards to accumulate on every poll.
function assertUniqueSiblingKeys(node: any): void {
  if (!node || typeof node !== 'object') return
  const flatten = (value: any): any[] => Array.isArray(value) ? value.flatMap(flatten) : value && typeof value === 'object' ? [value] : []
  const children = flatten(node.props?.children)
  const keys = children.map(child => child.key).filter(key => key !== null && key !== undefined)
  assert.equal(new Set(keys).size, keys.length, 'polling panes need distinct sibling identities')
  children.forEach(assertUniqueSiblingKeys)
}
for (const accounts of [[], [account('account-a')]]) {
  const noopComponent = () => null
  const host = harness('../GoogleWorkspaceConnector.tsx', {}, {
    '../../../lib/ipc': { api: { platform: 'browser' } }, '../../ui': { Button }, '../shared': { Section: noopComponent },
    './google/GoogleAccountSetup': { GoogleAccountSetup: noopComponent }, './google/GoogleActionReviews': { GoogleActionReviews: noopComponent },
    './google/GoogleCalendarReader': { GoogleCalendarReader: noopComponent }, './google/GoogleMailDrafts': { GoogleMailDrafts: noopComponent },
    './google/GoogleMailReader': { GoogleMailReader: noopComponent }, './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: { clientConfigured: true, accounts, reviews: [], recentActions: [] }, busy: '', accountErrors: {}, refresh: async () => {} }) },
  })
  for (let i = 0; i < 3; i++) assertUniqueSiblingKeys(host.render('GoogleWorkspaceConnector', {}))
  host.close()
}
for (const status of ['disconnected', 'needs-reauthorization']) {
  const fixtureAccount = { ...account('account-a'), status, lastError: 'Disconnected locally; Google revocation could not be confirmed.' }
  const noopComponent = () => null
  const host = harness('../GoogleWorkspaceConnector.tsx', {}, {
    '../../../lib/ipc': { api: { platform: 'browser' } }, '../../ui': { Button }, '../shared': { Section: noopComponent },
    './google/GoogleAccountSetup': { GoogleAccountSetup: noopComponent }, './google/GoogleActionReviews': { GoogleActionReviews: noopComponent },
    './google/GoogleCalendarReader': { GoogleCalendarReader: noopComponent }, './google/GoogleMailDrafts': { GoogleMailDrafts: noopComponent },
    './google/GoogleMailReader': { GoogleMailReader: noopComponent }, './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: { clientConfigured: true, accounts: [fixtureAccount], reviews: [], recentActions: [] }, busy: '', accountErrors: { 'account-a': 'Transient action feedback' }, refresh: async () => {} }) },
  })
  host.render('GoogleWorkspaceConnector', {})
  const node = host.render('GoogleWorkspaceConnector', {})
  assert.ok(find(node, (item) => item.props?.role === 'alert' && visibleText(item.props.children) === fixtureAccount.lastError), 'durable failed-revocation/reauthorization feedback remains visible after local disconnect')
  assert.ok(find(node, (item) => item.props?.role === 'alert' && visibleText(item.props.children) === 'Transient action feedback'), 'durable account feedback is independent from transient errors')
  host.close()
}
const calls: string[] = []
const setup = harness('GoogleAccountSetup.tsx', { openExternal: async () => {} })
const signIns: any[] = []
const setupProps = { snapshot: { clientConfigured: true, accounts: [], reviews: [], recentActions: [] }, busy: '', importClient: () => calls.push('import'), authorize: (value: unknown) => signIns.push(value), cancel: () => {} }
const setupNode = setup.render('GoogleAccountSetup', setupProps)
const sendToggle = find(setupNode, (item) => item.type === Toggle && item.props.label === 'Send reviewed emails')
assert.equal(sendToggle.props.checked, false, 'write capabilities start off')
assert.equal(find(setupNode, (item) => item.type === Toggle && item.props.label === 'Apply reviewed calendar changes').props.checked, false)
button(setupNode, 'Sign in with Google').props.onClick()
assert.deepEqual(signIns[0], ['gmail.read', 'calendar.read'])
setup.close()
const imports: string[] = []
for (const profile of [
  { accounts: [], busy: '', waiting: false, allowed: true },
  { accounts: [account('connected')], busy: '', waiting: false, allowed: false },
  { accounts: [{ ...account('reauth'), status: 'needs-reauthorization' }], busy: '', waiting: false, allowed: false },
  { accounts: [{ ...account('old'), status: 'disconnected' }], busy: '', waiting: false, allowed: true },
  { accounts: [], busy: 'draft', waiting: false, allowed: false },
  { accounts: [], busy: '', waiting: true, allowed: false },
]) {
  const replace = harness('GoogleAccountSetup.tsx', { openExternal: async () => {} })
  const before = imports.length
  const replacement = replace.render('GoogleAccountSetup', { ...setupProps, snapshot: { ...setupProps.snapshot, accounts: profile.accounts, ...(profile.waiting ? { authorization: { status: 'waiting' } } : {}) }, busy: profile.busy, importClient: () => imports.push('native-import') })
  const control = button(replacement, 'Replace Desktop client JSON')
  assert.ok(control, 'configured client retains a recovery/import control')
  assert.equal(control.props.disabled, !profile.allowed)
  control.props.onClick()
  assert.equal(imports.length - before, profile.allowed ? 1 : 0, 'replacement is guarded even when its disabled handler is invoked directly')
  replace.close()
}

let resolveOld!: (value: any) => void
const oldRead = new Promise((resolve) => { resolveOld = resolve })
const readerApi = { googleWorkspaceMessages: async (id: string) => { calls.push('search-' + id); return oldRead } }
const a = harness('GoogleMailReader.tsx', readerApi)
let node = a.render('GoogleMailReader', { account: account('account-a') })
assert.deepEqual(calls, [], 'mounting Gmail reader never makes a remote mutation or implicit search')
find(node, (item) => item.type === 'form').props.onSubmit({ preventDefault() {} })
assert.deepEqual(calls, ['search-account-a'])
a.close()
const b = harness('GoogleMailReader.tsx', readerApi)
resolveOld({ items: [{ id: 'old', subject: 'Old account confidential mail' }] })
await Promise.resolve(); await Promise.resolve()
assert.equal(find(a.render('GoogleMailReader', { account: account('account-a') }), (item) => item.props?.children === 'Old account confidential mail'), undefined, 'unmounted pane discards its own late result rather than retaining hidden private content')
node = b.render('GoogleMailReader', { account: account('account-b') })
assert.equal(find(node, (item) => item.props?.children === 'Old account confidential mail'), undefined, 'late old-account read never enters the selected account')
b.close()

const writes: any[] = []
const drafts = harness('GoogleMailDrafts.tsx', {
  googleWorkspaceDrafts: async () => [],
  googleWorkspaceSaveDraft: async (id: string, email: any) => { writes.push({ kind: 'local', id, email }); return { ...email, id: 'draft-one', accountId: id, localOnly: true, updatedAt: 0 } },
  googleWorkspacePrepareAction: async (id: string, action: any) => { writes.push({ kind: 'prepare', id, action }); return {} },
})
const props = { account: account('account-a'), refresh: async () => {} }
node = drafts.render('GoogleMailDrafts', props)
await Promise.resolve(); await Promise.resolve()
assert.equal(writes.length, 0, 'draft list mount performs no send or write')
input(node, 'Email to').props.onChange({ target: { value: 'recipient@example.com' } })
node = drafts.render('GoogleMailDrafts', props)
input(node, 'Email subject').props.onChange({ target: { value: 'Synthetic draft' } })
node = drafts.render('GoogleMailDrafts', props)
assert.equal(button(node, 'Review email before sending').props.disabled, true, 'read-only grant cannot prepare a send')
await button(node, 'Save local draft').props.onClick()
assert.equal(writes[0].kind, 'local')
assert.equal(writes[0].id, 'account-a')
assert.equal(writes.length, 1, 'saving a local draft never calls preparation or execution')
drafts.close()
const requests = harness('useGoogleAccountRequest.ts', {})
const scoped = requests.render('useGoogleAccountRequest', account('account-a'))
let resolveFirst!: (value: string) => void
const late = new Promise<string>((resolve) => { resolveFirst = resolve })
const applied: string[] = []
const old = scoped.run('old-search', () => late, (value: string) => applied.push(value))
await scoped.run('new-search', async () => 'new-result', (value: string) => applied.push(value))
resolveFirst('old-result'); await old
assert.deepEqual(applied, ['new-result'], 'an older search cannot overwrite a newer result within the same account')
requests.close()

const reviewCalls: string[] = []
const reviews = harness('GoogleActionReviews.tsx', {
  googleWorkspaceExecuteAction: async () => { reviewCalls.push('execute'); return { id: 'review-one', status: 'uncertain' } },
  googleWorkspaceCancelAction: async () => { reviewCalls.push('cancel'); return { id: 'review-one', status: 'failed' } },
})
const review = { id: 'review-one', accountId: 'account-a', accountEmail: 'account-a@example.com', revision: 1, digest: 'fixture-digest', expiresAt: Date.now() + 60_000, action: { kind: 'gmail.send', email: { to: ['recipient@example.com'], subject: 'Review fixture', text: 'Exact fixture body' } } }
const reviewProps = { reviews: [review], recentActions: [], refresh: async () => {} }
node = reviews.render('GoogleActionReviews', reviewProps)
assert.deepEqual(reviewCalls, [], 'rendering prepared reviews never executes them')
await button(node, 'Review and approve on this Mac').props.onClick()
assert.deepEqual(reviewCalls, ['execute'], 'one deliberate approval invokes one guarded execution')
node = reviews.render('GoogleActionReviews', { ...reviewProps, reviews: [], recentActions: [{ review, result: { id: 'review-one', status: 'uncertain' } }] })
assert.equal(button(node, 'Review and approve on this Mac'), undefined, 'settled uncertain results never expose an execution/retry button')
node = reviews.render('GoogleActionReviews', { ...reviewProps, reviews: [{ ...review, expiresAt: 1 }] })
assert.equal(button(node, 'Review and approve on this Mac').props.disabled, true)
await button(node, 'Review and approve on this Mac').props.onClick()
assert.deepEqual(reviewCalls, ['execute'], 'even a directly invoked expired handler cannot execute')
node = reviews.render('GoogleActionReviews', { ...reviewProps, reviews: [], recentActions: [{ review, result: { id: 'review-one', status: 'failed', cancelled: true } }] })
assert.ok(find(node, (item) => item.type === 'summary' && React.Children.toArray(item.props.children).includes('Cancelled by you')), 'durable owner cancellation is distinct from a failed send')
reviews.close()
console.log('Actual Google renderer read-only defaults, deferred cross-account read fence, explicit local draft save and no implicit sends passed')
