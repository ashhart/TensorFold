import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import * as edit from '../../lib/ticket-details'
import { fixtureText, reviewedLanguageImport } from '../preferences/language-test-fixture'

let slots: any[] = [], cursor = 0
const effects: Array<() => void> = []
let closed = 0, failLoad = false
const calls: any[][] = []
let resolve: (() => void) | undefined, reject: ((error: Error) => void) | undefined
const ticket: any = { id: 'ticket', projectId: 'project', title: 'Original', body: 'Full body\n'.repeat(60), type: 'task', priority: 'medium', effort: 'small', runState: null }
const state: any = {
  tickets: [ticket], projects: [{ id: 'project' }], archives: { tickets: [] },
  updateTicket: (id: string, patch: unknown) => { calls.push(['save', id, patch]); return new Promise<void>((yes, no) => { resolve = yes; reject = no }) },
  refreshArchives: async () => { if (failLoad) throw new Error('Synthetic archive load failed') },
  restoreItem: (...args: unknown[]) => { calls.push(['restore', ...args]); return new Promise<void>((yes, no) => { resolve = yes; reject = no }) },
}
const imports: Record<string, unknown> = {
  react: {
    useState: (value: any) => { const index = cursor++; if (!(index in slots)) slots[index] = typeof value === 'function' ? value() : value; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
    useRef: (value: any) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: value }; return slots[index] },
    useEffect: (run: () => void) => { const index = cursor++; if (!(index in slots)) { slots[index] = true; effects.push(run) } },
    useId: () => 'ticket-dialog',
  },
  'react/jsx-runtime': { jsx: (type: any, props: any) => React.createElement(type, props), jsxs: (type: any, props: any) => React.createElement(type, props), Fragment: React.Fragment },
  'react-dom': { createPortal: (node: unknown) => node },
  '../../lib/store': { useStore: Object.assign((select: (value: any) => unknown) => select(state), { getState: () => state }) },
  '../../lib/ticket-details': edit,
  '../../lib/store/shared/merge': { archiveOriginHostId: (item: any) => item.orchestratorHostId ?? item.orchestratorId ?? null },
  '../../lib/store/shared/proxy-state': { projectedProjectHostId: (project: any) => project.executionTarget === 'orchestrator' ? project.orchestratorId ?? null : null },
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => fixtureText },
  '../navigation/useAccessibleDialog': { useAccessibleDialog() {} },
  '../../lib/icons': { Icon: () => null },
  '../OriginalDiagnostic': { OriginalDiagnostic: () => null },
}
const loaded = { exports: {} as any }
const output = ts.transpileModule(readFileSync(path.join(__dirname, 'TicketDetails.tsx'), 'utf8') + '\nexports.TicketEditor = TicketEditor; exports.TicketArchive = TicketArchive;', {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText
new Function('require', 'module', 'exports', output)((id: string) => {
  const reviewed = reviewedLanguageImport(id, imports)
  if (reviewed.matched) return reviewed.value
  assert.ok(id in imports, `Reviewed ticket dialog import ${id}`)
  return imports[id]
}, loaded, loaded.exports)
const reset = () => { slots = []; effects.length = 0; calls.length = 0; closed = 0 }
const render = (component: string, props: unknown) => { cursor = 0; const node = loaded.exports[component](props); effects.splice(0).forEach((run) => run()); return node }
function find(node: any, type: unknown): any[] {
  if (!node || typeof node !== 'object') return []
  return [...(node.type === type ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => find(child, type))]
}
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve() }
const props = { ticket, onClose: () => closed++ }
let node = render('TicketEditor', props)
assert.equal(find(node, 'textarea')[0].props.value, ticket.body, 'the editor exposes all text without truncation')
find(node, 'input')[0].props.onChange({ target: { value: 'Revised' } })
find(node, 'textarea')[0].props.onChange({ target: { value: 'Full revised description' } })
for (const [index, value] of ['bug', 'urgent', 'large'].entries()) find(node, 'select')[index].props.onChange({ target: { value } })
node = render('TicketEditor', props)
const submit = () => find(node, 'form')[0].props.onSubmit({ preventDefault() {} })
submit(); submit()
assert.deepEqual(calls, [['save', 'ticket', { title: 'Revised', body: 'Full revised description', type: 'bug', priority: 'urgent', effort: 'large' }]], 'save is locked before a second click can enqueue another mutation')
reject!(new Error('Synthetic save failed')); await settle()
node = render('TicketEditor', props)
assert.equal(find(node, 'textarea')[0].props.value, 'Full revised description', 'failed saves retain the draft')
assert.equal(closed, 0)
assert.equal(find(node, 'p').some((item) => item.props.role === 'alert'), true)
submit(); resolve!(); await settle()
assert.equal(closed, 1)

reset()
node = render('TicketEditor', { ...props, ticket: { ...ticket, runState: 'running' } })
assert.equal(find(node, 'textarea')[0].props.readOnly, true)
assert.equal(find(node, 'select').every((item) => item.props.disabled), true)
find(node, 'form')[0].props.onSubmit({ preventDefault() {} })
assert.equal(calls.length, 0, 'active work remains readable without allowing edits')

reset()
state.archives.tickets = [{ ...ticket, id: 'archived', archivedAt: 1 }, { ...ticket, id: 'other-project', projectId: 'other', archivedAt: 1 }, { ...ticket, id: 'remote', orchestratorHostId: 'remote-host', archivedAt: 1 }]
node = render('TicketArchive', { projectId: 'project', onClose() {} }); await settle()
node = render('TicketArchive', { projectId: 'project', onClose() {} })
assert.equal(find(node, 'section').length, 1, 'archives are filtered by both project and origin')
const restore = find(node, 'section')[0]
assert.equal(find(restore, 'p')[0].props.children, ticket.body)
find(restore, 'button')[0].props.onClick(); find(restore, 'button')[0].props.onClick()
assert.deepEqual(calls, [['restore', 'ticket', 'archived', null]], 'restore passes explicit local ownership and blocks duplicate requests')
resolve!(); await settle()
reset()
state.projects = [{ id: 'project', executionTarget: 'orchestrator', orchestratorId: 'remote-host' }]
node = render('TicketArchive', { projectId: 'project', onClose() {} }); await settle()
node = render('TicketArchive', { projectId: 'project', onClose() {} })
find(find(node, 'section')[0], 'button')[0].props.onClick()
assert.deepEqual(calls, [['restore', 'ticket', 'remote', 'remote-host']], 'remote restores target the owning host')
reject!(new Error('Synthetic restore failed')); await settle()
node = render('TicketArchive', { projectId: 'project', onClose() {} })
assert.equal(find(node, 'p').some((item) => item.props.role === 'alert'), true)
assert.equal(find(node, 'section').length, 1, 'failed restores retain the archived entry')
console.log('Ticket dialogs cover full-text edits, settings, duplicate saves, retained errors/drafts, active runs, scoped archives and local/remote restore routing')
