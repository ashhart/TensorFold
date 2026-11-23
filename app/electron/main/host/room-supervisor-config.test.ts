import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { ROOM_COLLABORATION_MODES } from '../../agent/room-collaboration'
import { assertRoomConfigUpdateAllowed, assertRoomSeatRoutesConfigured, pickRoomConfigPatch } from './room-ticket-patch-policy'

function handlerSource(file: string, match: (node: ts.Node) => boolean): string {
  const tree = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.ES2022, true)
  const found: ts.Node[] = []
  const visit = (node: ts.Node) => { if (match(node)) found.push(node); ts.forEachChild(node, visit) }
  visit(tree)
  assert.equal(found.length, 1, 'exercise exactly one actual production handler')
  const node = ts.isCallExpression(found[0]) ? found[0].arguments[1] : found[0]
  return ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, tree)
}
const local = handlerSource(path.join(__dirname, 'workspace-runtime-ipc.ts'), (node) => ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'handle' && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === 'rooms:update')
const remote = handlerSource(path.join(__dirname, 'remote-host-api.ts'), (node) => ts.isFunctionDeclaration(node) && node.name?.text === 'updateRoomOnThisHost')

for (const [surface, source] of [['local', local], ['remote', remote]]) {
  const room: any = { id: 'room', projectId: 'project', state: 'ready', seats: [], events: [], collaborationMode: 'balanced', deliberationEffort: 'deep', workerBudget: { maxActive: 1, maxStarts: 2 } }
  const store = { rooms: [room] }, aborters = new Map(), published: any[] = []
  let persisted = 0
  const publish = (event: any) => published.push(structuredClone(event))
  const bindings = {
    store, pickRoomConfigPatch, assertRoomConfigUpdateAllowed, assertRoomSeatRoutesConfigured,
    requireSecurityRole: (role: string) => assert.equal(role, 'operator'), roomAborters: aborters,
    persistRooms: () => persisted++, publishRemote: publish, sendToRenderer() {},
    requireActiveRoomOnThisHost: () => { if (room.archivedAt) throw new Error('Restore the archived room before updating it.'); return room },
    deps: { roomAborters: () => aborters, remoteServer: () => ({ publish }) },
    remoteMutationError: (message: string, status: number) => Object.assign(new Error(message), { status }),
  }
  const code = surface === 'local' ? `const handler = ${source};` : `${source}; const handler = updateRoomOnThisHost;`
  const compiled = ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  const handler = new Function(...Object.keys(bindings), `${compiled}; return handler;`)(...Object.values(bindings))
  const invoke = (patch: unknown) => surface === 'local' ? handler(null, { id: room.id, patch }) : handler(room.id, patch)

  const dropdown = await invoke({ collaborationMode: 'supervisor' })
  assert.equal(dropdown.collaborationMode, 'supervisor', `${surface}: the Supervisor dropdown selection must be accepted`)
  assert.equal(dropdown.deliberationEffort, 'deep', 'a dropdown change preserves the existing effort')
  assert.deepEqual(dropdown.workerBudget, { maxActive: 1, maxStarts: 2 })
  assert.equal(persisted, 1)
  assert.equal(published[0].kind, 'room')
  assert.equal(published[0].data.collaborationMode, 'supervisor')
  const preset = { collaborationMode: 'supervisor', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 4 } }
  const updated = await invoke(preset)
  for (const [field, value] of Object.entries(preset)) assert.deepEqual(updated[field], value, 'Supervised build saves its complete configuration')
  for (const { id } of ROOM_COLLABORATION_MODES) assert.equal((await invoke({ collaborationMode: id })).collaborationMode, id)
  const saved = persisted
  for (const collaborationMode of ['bogus', ' supervisor', null, 0, {}]) await assert.rejects(invoke({ collaborationMode }), /collaborationMode has an invalid value/)
  await assert.rejects(invoke({ collaborationMode: 'supervisor', state: 'complete' }), /state cannot be changed/)
  for (const state of ['queued', 'running', 'ready']) {
    room.state = state
    if (state === 'ready') aborters.set(room.id, {})
    await assert.rejects(invoke(preset), /Stop the active room run/)
    aborters.clear()
  }
  room.archivedAt = 2
  await assert.rejects(invoke(preset), /archived/)
  assert.equal(persisted, saved, 'invalid and active/archived updates never persist')
  assert.equal(published.length, saved, 'rejected updates never publish changes')
}
console.log('Local/remote Supervisor selections and Supervised build presets persist; supported modes and invalid/live/archive guards pass')
