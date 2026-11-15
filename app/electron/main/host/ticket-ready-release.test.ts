import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { pickTicketUserPatch, assertTicketUserUpdateAllowed } from './room-ticket-patch-policy'
import { assertTicketRelations } from './ticket-helpers'
import { normalizeTicketExecution } from '../../agent/tickets'

function declaration(file: string, match: (node: ts.Node) => boolean): string {
  const tree = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.ES2022, true)
  const found: ts.Node[] = []
  const visit = (node: ts.Node) => { if (match(node)) found.push(node); ts.forEachChild(node, visit) }
  visit(tree)
  assert.equal(found.length, 1, 'extract the actual production handler exactly once')
  const node = ts.isCallExpression(found[0]) ? found[0].arguments[1] : found[0]
  return ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, tree)
}
const local = declaration(path.join(__dirname, 'workspace-runtime-ipc.ts'), (node) => ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'handle' && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === 'tickets:update')
const remote = declaration(path.join(__dirname, 'remote-host-api.ts'), (node) => ts.isFunctionDeclaration(node) && node.name?.text === 'updateTicketOnThisHost')

for (const [surface, source] of [['local', local], ['remote', remote]]) {
  const ticket: any = { id: 'ticket', projectId: 'project', roomId: 'room', title: 'Synthetic', body: '', type: 'task', status: 'in_progress', priority: 'medium', effort: 'small', tags: [], assignee: 'Builder', threadId: 'thread', runState: null, evidence: [], createdAt: 1, updatedAt: 1 }
  const store: any = { tickets: [ticket], projects: [{ id: 'project' }], rooms: [{ id: 'room', projectId: 'project' }], threads: [{ id: 'thread', projectId: 'project' }] }
  const aborters = new Map(), published: unknown[] = []
  let starting = false, persisted = 0
  const bindings = {
    store, pickTicketUserPatch, assertTicketUserUpdateAllowed, assertTicketRelations, normalizeTicketExecution,
    requireSecurityRole() {}, aborters, isTicketStarting: () => starting,
    normalizeStringList: (value: string[]) => value,
    persistTickets: () => persisted++, emitTicket: (value: unknown) => published.push(structuredClone(value)),
    deps: { aborters: () => aborters, ticketWorkerRuntime: () => ({ isTicketStarting: () => starting }), emitTicket: (value: unknown) => published.push(structuredClone(value)) },
    validateTicketOwnershipOnThisHost: (projectId: string, roomId: string, options: any) => assertTicketRelations(store, projectId, roomId, options),
    remoteMutationError: (message: string, status = 409) => Object.assign(new Error(message), { status }),
  }
  const code = surface === 'local' ? `const handler = ${source};` : `${source}; const handler = updateTicketOnThisHost;`
  const compiled = ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  const handler = new Function(...Object.keys(bindings), `${compiled}; return handler;`)(...Object.values(bindings))
  const invoke = (patch: unknown) => surface === 'local' ? handler(null, { id: ticket.id, patch }) : handler(ticket.id, patch)

  const next = await invoke({ status: 'ready' })
  assert.equal(next.assignee, null, `${surface}: returning a parked ticket to Ready must release its previous owner`)
  assert.equal(next.threadId, 'thread', 'release preserves the existing worker chat/history')
  assert.equal(persisted, 1)
  assert.equal((published[0] as any).assignee, null, 'the released claim is published to clients')
  ticket.assignee = 'Reviewer'
  await invoke({ priority: 'high' })
  assert.equal(ticket.assignee, 'Reviewer', 'ordinary edits never change ticket ownership')
  for (const runState of ['queued', 'running', null]) {
    ticket.runState = runState
    if (runState === null) aborters.set('thread', {})
    await assert.rejects(invoke({ status: 'ready' }), /Stop the active ticket/)
    assert.equal(ticket.assignee, 'Reviewer', 'a rejected transition must not clear a live claim')
    aborters.clear()
  }
  ticket.runState = null; starting = true
  await assert.rejects(invoke({ status: 'ready' }), /Stop the active ticket/)
  starting = false
  for (const patch of [{ assignee: 'Another' }, { status: 'ready', assignee: null }]) {
    await assert.rejects(invoke(patch), /assignee cannot be changed/, 'clients cannot forge host-owned ownership fields')
  }
  ticket.archivedAt = 2
  await assert.rejects(invoke({ status: 'ready' }), /archived/)
  assert.equal(ticket.assignee, 'Reviewer')
}
console.log('Local/remote Ready transitions release parked claims, preserve history and retain active/startup/archive/privileged-field guards')
