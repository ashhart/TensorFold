import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { reviewedLanguageImport } from '../../../preferences/language-test-fixture'
import { GoogleWorkspaceService } from '../../../../../electron/agent/google-workspace'
import { requestedScopes } from '../../../../../electron/agent/google-workspace/scopes'
import type { GoogleWorkspaceAccount, GoogleWorkspaceSnapshot } from '../../../../../shared/google-workspace'

const Button = (_props: any) => null, Toggle = (_props: any) => null, Field = (_props: any) => null, Input = (_props: any) => null
const Empty = (_props: any) => null
const MailReader = (_props: any) => null, CalendarReader = (_props: any) => null
const MailDrafts = (_props: any) => null, AccountSetup = (_props: any) => null
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }

// Exercise the production hooks and components at the same isolated seam as
// GoogleRendererFlows; no browser account, OAuth token or network is consulted.
function harness(file: string, api: any, extra: Record<string, unknown> = {}) {
  const slots: any[] = [], cleanups = new Map<number, () => void>()
  let cursor = 0, effects: Array<() => void> = []
  const react = {
    useState(initial: any) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], (next: any) => { slots[i] = typeof next === 'function' ? next(slots[i]) : next }] },
    useRef(initial: any) { const i = cursor++; return slots[i] ??= { current: initial } },
    useCallback(fn: any, deps: any[]) { const i = cursor++; if (!slots[i] || deps.some((v, j) => v !== slots[i].deps[j])) slots[i] = { fn, deps }; return slots[i].fn },
    useEffect(work: () => any, deps: any[]) { const i = cursor++; if (!slots[i] || deps.some((v, j) => v !== slots[i][j])) { slots[i] = deps; effects.push(() => { cleanups.get(i)?.(); const cleanup = work(); if (typeof cleanup === 'function') cleanups.set(i, cleanup) }) } },
  }
  const imports: Record<string, any> = {
    react, 'react/jsx-runtime': jsx, '../../../../lib/ipc': { api }, '../../../lib/ipc': { api },
    '../../../ui': { Button, Toggle, Field, Input }, '../../ui': { Button }, '../shared': { Section: Empty },
    './google/GoogleAccountSetup': { GoogleAccountSetup: AccountSetup }, './google/GoogleActionReviews': { GoogleActionReviews: Empty },
    './google/GoogleCalendarReader': { GoogleCalendarReader: CalendarReader }, './google/GoogleMailDrafts': { GoogleMailDrafts: MailDrafts },
    './google/GoogleMailReader': { GoogleMailReader: MailReader }, './google-setup-error': { googleClientImportError: () => 'Client import failed.' },
    ...extra,
  }
  const source = ts.transpileModule(readFileSync(path.join(__dirname, file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  const loaded = { exports: {} as any }
  new Function('require', 'module', 'exports', source)((name: string) => {
    const reviewed = reviewedLanguageImport(name, imports)
    if (reviewed.matched) return reviewed.value
    assert.ok(name in imports, `Reviewed recovery fixture import ${name}`)
    return imports[name]
  }, loaded, loaded.exports)
  return {
    render(name: string, ...args: any[]) { cursor = 0; effects = []; const result = loaded.exports[name](...args); effects.forEach(work => work()); return result },
    close() { cleanups.forEach(cleanup => cleanup()); cleanups.clear() },
  }
}
const flush = () => new Promise<void>(resolve => setImmediate(resolve))
const account = (id: string, status: GoogleWorkspaceAccount['status'] = 'connected'): GoogleWorkspaceAccount => ({ id, email: `${id}@example.com`, revision: 1, capabilities: ['gmail.read', 'calendar.read'], status, connectedAt: 1 })
const snapshot = (accounts: GoogleWorkspaceAccount[] = []): GoogleWorkspaceSnapshot => ({ clientConfigured: true, accounts, reviews: [], recentActions: [] })
function find(node: any, match: (item: any) => boolean): any {
  if (!node || typeof node !== 'object') return undefined
  if (match(node)) return node
  for (const child of React.Children.toArray(node.props?.children)) { const result = find(child, match); if (result) return result }
}
const text = (node: any): string => React.Children.toArray(node).map((child: any) => child && typeof child === 'object' ? text(child.props?.children) : String(child ?? '')).join(' ')
const button = (node: any, title: string) => find(node, item => item.type === Button && item.props.children === title)

for (const accounts of [[], [account('first')], [account('first'), account('second')]]) {
  const expected = snapshot(accounts)
  const hook = harness('useGoogleWorkspace.ts', { platform: 'darwin', googleWorkspaceStatus: async () => expected })
  assert.equal(hook.render('useGoogleWorkspace').snapshot, null, 'loading is not an empty account result')
  await flush()
  const state = hook.render('useGoogleWorkspace')
  assert.deepEqual(state.snapshot.accounts, accounts, 'zero, one and multiple identities survive discovery')
  assert.equal(state.error, '')
  hook.close()
}

let unavailable = true
const discovery = harness('useGoogleWorkspace.ts', {
  platform: 'darwin',
  googleWorkspaceStatus: async () => { if (unavailable) throw new Error('synthetic-private-token-must-not-appear'); return snapshot() },
})
discovery.render('useGoogleWorkspace'); await flush()
let state = discovery.render('useGoogleWorkspace')
assert.equal(state.snapshot, null, 'a discovery/storage failure must not become no accounts')
assert.match(state.error, /could not be read/)
assert.ok(!state.error.includes('synthetic-private-token'))
unavailable = false
assert.equal(await state.action('refresh', state.refresh), true, 'Retry restores discovery')
state = discovery.render('useGoogleWorkspace')
assert.deepEqual(state.snapshot.accounts, [])
assert.equal(state.error, '')
discovery.close()

// Reproduce the stale Connected badge through a real revoked-refresh failure,
// rather than inventing a renderer-only account status transition.
const now = 1_800_000_000_000
let transportCalls = 0
let privateState: any = { version: 1, client: { clientId: '123456789012-synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' }, accounts: [{ ...account('first'), subject: 'synthetic-subject', tokens: { accessToken: 'synthetic-expired', refreshToken: 'synthetic-refresh', expiresAt: now, scopes: requestedScopes(['gmail.read', 'calendar.read']) } }], drafts: [], actions: [] }
const service = new GoogleWorkspaceService({
  now: () => now,
  store: { read: async () => structuredClone(privateState), write: async next => { privateState = structuredClone(next) } },
  openExternal: async () => { throw new Error('No browser sign-in in this fixture') },
  lookup: async () => [{ address: '8.8.8.8', family: 4 }],
  fetchImpl: (async (url: any) => {
    assert.equal(String(url), 'https://oauth2.googleapis.com/token')
    transportCalls++
    return new Response(JSON.stringify({ error: 'invalid_grant', error_description: 'synthetic-private-token-must-not-appear' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
  }) as typeof fetch,
})
const workspace = harness('useGoogleWorkspace.ts', { platform: 'darwin', googleWorkspaceStatus: () => service.snapshot() })
workspace.render('useGoogleWorkspace'); await flush()
state = workspace.render('useGoogleWorkspace')
assert.equal(state.snapshot.accounts[0].status, 'connected')
const requests = harness('useGoogleAccountRequest.ts', {})
const request = requests.render('useGoogleAccountRequest', state.snapshot.accounts[0], state.refresh)
assert.equal(await request.run('search', () => service.listMessages('first', { query: 'is:unread' }), () => assert.fail('revoked request cannot apply mail')), false)
assert.equal(transportCalls, 1, 'the injected synthetic token exchange handled the request without Google networking')
assert.equal((await service.snapshot()).accounts[0].status, 'needs-reauthorization', 'host removes expired credentials')
assert.equal(workspace.render('useGoogleWorkspace').snapshot.accounts[0].status, 'needs-reauthorization', 'failed read must refresh the GUI account state so Reconnect is shown immediately')
requests.close(); workspace.close(); service.dispose()

let refreshCalls = 0, rejectOld!: (error: Error) => void
const lateFailure = new Promise<void>((_resolve, reject) => { rejectOld = reject })
const raced = harness('useGoogleAccountRequest.ts', {})
const active = raced.render('useGoogleAccountRequest', account('first'), async () => { refreshCalls++ })
const oldRead = active.run('old', () => lateFailure, () => assert.fail('failed read cannot apply'))
await active.run('new', async () => 'current', () => {})
rejectOld(new Error('synthetic-private-token-must-not-appear')); await oldRead
assert.equal(refreshCalls, 0, 'superseded failures cannot refresh or replace the current account pane')
raced.close()

const failedRefresh = harness('useGoogleAccountRequest.ts', {})
const failing = failedRefresh.render('useGoogleAccountRequest', account('first'), async () => { throw new Error('synthetic-private-store-path') })
await failing.run('search', async () => { throw new Error('synthetic-private-token') }, () => {})
const failedState = failedRefresh.render('useGoogleAccountRequest', account('first'))
assert.match(failedState.error, /refresh the account status; use Refresh/)
assert.ok(!failedState.error.includes('synthetic-private'))
assert.equal(failedState.busy, '')
failedRefresh.close()

const retainedRequest = harness('useGoogleAccountRequest.ts', {})
const renewedAccount = { ...account('first', 'needs-reauthorization'), revision: 2 }
const reconcile = async () => { retainedRequest.render('useGoogleAccountRequest', renewedAccount, reconcile) }
const retained = retainedRequest.render('useGoogleAccountRequest', account('first'), reconcile)
await retained.run('save', async () => { throw new Error('Synthetic account state changed') }, () => {})
assert.equal(retainedRequest.render('useGoogleAccountRequest', renewedAccount, reconcile).busy, '', 'a retained pane cannot stay busy after its request belongs to an older account revision')
retainedRequest.close()

for (const status of ['connected', 'needs-reauthorization', 'disconnected'] as const) {
  const current = snapshot([account('first', status), account('second')])
  const pane = harness('../GoogleWorkspaceConnector.tsx', { platform: 'darwin' }, { './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: current, busy: '', accountErrors: {}, refresh: async () => {} }) } })
  pane.render('GoogleWorkspaceConnector')
  let node = pane.render('GoogleWorkspaceConnector')
  const select = find(node, item => item.type === 'select' && item.props['aria-label'] === 'Google account')
  assert.equal(React.Children.count(select.props.children), 3, 'both saved accounts and an add/sign-in route remain selectable')
  if (status !== 'connected') assert.match(text(node), /Reconnect with Google/i)
  select.props.onChange({ target: { value: 'second' } })
  node = pane.render('GoogleWorkspaceConnector')
  assert.equal(find(node, item => item.type === 'select' && item.props['aria-label'] === 'Google account').props.value, 'second')
  assert.equal(find(node, item => item.type === MailReader).props.account.id, 'second')
  assert.equal(typeof find(node, item => item.type === MailReader).props.refresh, 'function', 'mail failures can reconcile host account status')
  assert.equal(typeof find(node, item => item.type === CalendarReader).props.refresh, 'function', 'calendar failures can reconcile host account status')
  pane.close()
}

const emptyPane = harness('../GoogleWorkspaceConnector.tsx', { platform: 'darwin' }, { './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: snapshot(), busy: '', accountErrors: {}, refresh: async () => {} }) } })
emptyPane.render('GoogleWorkspaceConnector')
assert.match(text(emptyPane.render('GoogleWorkspaceConnector')), /No Google accounts are connected/, 'successful empty discovery has an explanation, not just an empty selector')
emptyPane.close()

const errorPane = harness('../GoogleWorkspaceConnector.tsx', { platform: 'darwin' }, { './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: null, error: 'Google connections could not be read; check the local credential store, then retry.', busy: '', refresh: async () => {}, action: async () => true }) } })
const errorNode = errorPane.render('GoogleWorkspaceConnector')
assert.match(text(errorNode), /Google account status is unavailable/)
assert.ok(!text(errorNode).includes('No Google accounts are connected'))
assert.ok(button(errorNode, 'Retry'), 'discovery errors have a recovery action instead of an empty selector')
errorPane.close()

let currentSnapshot = snapshot([account('first')])
const dirtyPane = harness('../GoogleWorkspaceConnector.tsx', { platform: 'darwin' }, { './google/useGoogleWorkspace': { useGoogleWorkspace: () => ({ snapshot: currentSnapshot, busy: '', accountErrors: {}, refresh: async () => {} }) } })
dirtyPane.render('GoogleWorkspaceConnector')
let dirtyNode = dirtyPane.render('GoogleWorkspaceConnector')
const originalDraft = find(dirtyNode, item => item.type === MailDrafts)
const originalToolsKey = find(dirtyNode, item => item.type === 'div' && String(item.key).includes('tools')).key
originalDraft.props.onDirtyChange(true)
dirtyPane.render('GoogleWorkspaceConnector')
currentSnapshot = snapshot([{ ...account('first', 'needs-reauthorization'), revision: 2 }])
dirtyNode = dirtyPane.render('GoogleWorkspaceConnector')
assert.equal(find(dirtyNode, item => item.type === MailDrafts)?.key, originalDraft.key, 'revocation retains the same unsaved draft component')
assert.equal(find(dirtyNode, item => item.type === 'div' && String(item.key).includes('tools')).key, originalToolsKey, 'the parent subtree also retains its account identity across revisions')
assert.match(text(dirtyNode), /unsaved email edits remain on this page/)
const reconnect = find(dirtyNode, item => item.type === AccountSetup)
assert.equal(reconnect.props.busy, '', 'dirty drafts must not prevent same-account reauthorization')
assert.equal(reconnect.props.permissionsLocked, true, 'reconnection preserves the selected grant while edits are unsaved')
assert.equal(find(dirtyNode, item => item.type === 'select' && item.props['aria-label'] === 'Google account').props.disabled, true, 'unsaved drafts still block account switching')
dirtyPane.close()

let draftReads = 0
const localDraft = harness('GoogleMailDrafts.tsx', { googleWorkspaceDrafts: async () => { draftReads++; return [] } }, { './useGoogleAccountRequest': { useGoogleAccountRequest: () => ({ busy: '', error: '', run: async (_name: string, work: any, apply: any) => { apply(await work()); return true } }) } })
const draftProps = { account: account('first'), refresh: async () => {}, onDirtyChange: () => {} }
let localNode = localDraft.render('GoogleMailDrafts', draftProps)
find(localNode, item => item.type === Input && item.props['aria-label'] === 'Email to').props.onChange({ target: { value: 'recipient@example.com' } })
localNode = localDraft.render('GoogleMailDrafts', draftProps)
find(localNode, item => item.type === Input && item.props['aria-label'] === 'Email subject').props.onChange({ target: { value: 'Unsaved fixture draft' } })
await flush()
localNode = localDraft.render('GoogleMailDrafts', { ...draftProps, account: { ...account('first', 'needs-reauthorization'), revision: 2 } })
assert.equal(find(localNode, item => item.type === Input && item.props['aria-label'] === 'Email subject').props.value, 'Unsaved fixture draft')
assert.equal(button(localNode, 'Save local draft').props.disabled, true, 'revoked sessions cannot save or prepare work')
assert.equal(draftReads, 1, 'retained disconnected drafts do not trigger another authenticated read')
localNode = localDraft.render('GoogleMailDrafts', { ...draftProps, account: { ...account('first'), revision: 3 } })
assert.equal(find(localNode, item => item.type === Input && item.props['aria-label'] === 'Email subject').props.value, 'Unsaved fixture draft', 'same-account reconnect preserves the edits')
assert.equal(button(localNode, 'Save local draft').props.disabled, false)
localDraft.close()

let preparations = 0
const preparedDraft = harness('GoogleMailDrafts.tsx', {
  googleWorkspaceDrafts: async () => [],
  googleWorkspacePrepareAction: async () => { preparations++; return {} },
}, { './useGoogleAccountRequest': { useGoogleAccountRequest: () => ({ busy: '', error: '', run: async (_name: string, work: any, apply: any) => { apply(await work()); return true } }) } })
const sendAccount: GoogleWorkspaceAccount = { ...account('first'), capabilities: ['gmail.read', 'gmail.send'] }
const preparedProps = { account: sendAccount, refresh: async () => {} }
let preparedNode = preparedDraft.render('GoogleMailDrafts', preparedProps)
find(preparedNode, item => item.type === Input && item.props['aria-label'] === 'Email to').props.onChange({ target: { value: 'recipient@example.com' } })
preparedNode = preparedDraft.render('GoogleMailDrafts', preparedProps)
find(preparedNode, item => item.type === Input && item.props['aria-label'] === 'Email subject').props.onChange({ target: { value: 'Review after reconnect' } })
preparedNode = preparedDraft.render('GoogleMailDrafts', preparedProps)
await button(preparedNode, 'Review email before sending').props.onClick()
assert.equal(button(preparedDraft.render('GoogleMailDrafts', preparedProps), 'Review email before sending').props.disabled, true)
preparedDraft.render('GoogleMailDrafts', { ...preparedProps, account: { ...sendAccount, status: 'needs-reauthorization', revision: 2 } })
const reconnectedProps = { ...preparedProps, account: { ...sendAccount, revision: 3 } }
preparedDraft.render('GoogleMailDrafts', reconnectedProps)
preparedNode = preparedDraft.render('GoogleMailDrafts', reconnectedProps)
assert.equal(button(preparedNode, 'Review email before sending').props.disabled, false, 'reconnect invalidates the old review fingerprint without losing the draft')
await button(preparedNode, 'Review email before sending').props.onClick()
assert.equal(preparations, 2, 'the retained draft can request a fresh owner review after reconnect')
preparedDraft.close()

let authorizeCalls = 0
const setup = harness('GoogleAccountSetup.tsx', {})
const setupProps = { snapshot: snapshot(), busy: '', importClient: () => {}, authorize: () => { authorizeCalls++ }, cancel: () => {} }
button(setup.render('GoogleAccountSetup', setupProps), 'Sign in with Google').props.onClick()
assert.equal(authorizeCalls, 1, 'manual add uses the existing browser OAuth path')
assert.ok(!find(setup.render('GoogleAccountSetup', setupProps), item => item.type === 'input' && item.props.type === 'password'), 'setup never asks for a Google password')
let imported = false
const initialSetup = setup.render('GoogleAccountSetup', { ...setupProps, snapshot: { ...snapshot(), clientConfigured: false }, importClient: () => { imported = true } })
assert.equal(button(initialSetup, 'Sign in with Google'), undefined, 'Google client setup is an explicit prerequisite')
button(initialSetup, 'Import Desktop client JSON').props.onClick()
assert.equal(imported, true)
const reconnectNode = setup.render('GoogleAccountSetup', { ...setupProps, account: account('first', 'needs-reauthorization'), permissionsLocked: true })
assert.equal(button(reconnectNode, 'Reconnect with selected access').props.disabled, false)
assert.equal(find(reconnectNode, item => item.type === Toggle).props.disabled, true)
setup.close()
console.log('Google account discovery, revoked-session recovery, account selection and sign-in coverage passed')
