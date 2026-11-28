import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import clsx from 'clsx'
import * as roomModes from '../../lib/room-modes'
import type { Room, RoomEventItem, Ticket } from '../../lib/types'

let states: any[] = [], cursor = 0
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
function load(name: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  const output = ts.transpileModule(readFileSync(path.join(__dirname, name), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => {
    const reviewed = reviewedLanguageImport(id, imports)
    if (reviewed.matched) return reviewed.value
    assert.ok(id in imports, `Reviewed import ${id}`)
    return imports[id]
  }, module, module.exports)
  return module.exports
}

const helperImports: Record<string, unknown> = {
  '../../lib/room-modes': roomModes, '../../lib/format': { relativeTime: () => '' },
  '../../lib/policy': { allowedProviders: () => [], projectDefaultModel: () => null }, '../../lib/provider-accounts': { providerAccountChoices: () => [] }, 'react/jsx-runtime': jsx,
}
const roomHelpers = load('helpers.tsx', helperImports)
// The panel calls the real RoomTimeline snippet helper, so load the component instead of copying it.
const timeline = load('RoomTimeline.tsx', {
  clsx: { __esModule: true, default: clsx }, '../../lib/format': { basename: (value: string) => value },
  '../../lib/icons': { Icon: () => null }, '../../lib/fileicon': { FileGlyph: () => null },
  './helpers': roomHelpers, 'react/jsx-runtime': jsx,
})
const imports: Record<string, unknown> = {
  react: { useState: (initial: any) => { const index = cursor++; if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial; return [states[index], (value: any) => { states[index] = typeof value === 'function' ? value(states[index]) : value }] } },
  'react/jsx-runtime': jsx, clsx: { __esModule: true, default: clsx }, '../../lib/icons': { Icon: () => null },
  '../../lib/ipc': { api: { mergeRoomWorktree: async () => ({ ok: true, status: 'nothing', merged: [], baseBranch: 'main', blockers: [], conflict: null, remaining: [] }) } },
  './helpers': roomHelpers, './RoomTimeline': { roomEventSnippet: timeline.roomEventSnippet },
}
const plans = load('RoomPlans.tsx', imports)

const room = { id: 'room-1', projectId: 'p', name: 'Room', finalText: '', state: 'ready', seats: [], problem: '', memory: [] } as unknown as Room
const ticket = (order: number): Ticket => ({ id: `t${order}`, projectId: 'p', roomId: room.id, title: `Task ${order}`, body: '', type: 'task', status: 'completed', priority: 'medium', effort: 'small', tags: [], parentId: null, assignee: null, threadId: null, runState: null, order, createdAt: 1, updatedAt: 1 }) as Ticket
const completed = (count: number) => Array.from({ length: count }, (_, index) => ticket(index + 1))
const event = (kind: string, text: string, createdAt: number): RoomEventItem => ({ id: `${kind}-${createdAt}`, tick: createdAt, kind, seat: kind === 'final' ? 'planner' : 'moderator', target: 'builder', text, createdAt })

function render(events: RoomEventItem[], options: { room?: Room; tickets?: Ticket[]; leftToWork?: Ticket[] } = {}) {
  states = []; cursor = 0
  return plans.RoomFinalGatePanel({ tickets: options.tickets ?? completed(8), leftToWork: options.leftToWork ?? [], conflictBlockers: [], room: options.room ?? room, events })
}
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function text(node: any): string {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(text).join(' ')
  return typeof node === 'object' && node.props ? text(node.props.children) : ''
}
const mergeButton = (node: any) => all(node, (entry) => entry.type === 'button' && React.Children.toArray(entry.props.children).includes('Merge worktree'))[0]
function rowIcon(node: any, label: string): any {
  const cell = all(node, (entry) => React.Children.toArray(entry.props?.children).includes(label))[0]
  return React.Children.toArray(cell.props.children)[0]
}
// The header icon is the first check/shield Icon in document order; the row icons follow it.
const panelIcon = (node: any) => all(node, (entry) => entry.props?.name === 'check' || entry.props?.name === 'shield')[0]
function assertPanel(node: any, green: boolean) {
  assert.equal(panelIcon(node).props.name, green ? 'check' : 'shield', green ? 'the panel is green' : 'the panel is not green')
  const copy = green ? 'The room has enough signal to finalize.' : 'The room still has gates to satisfy.'
  assert.ok(text(node).includes(copy), `the panel copy reads "${copy}"`)
  assert.ok(!text(node).includes(green ? 'The room still has gates to satisfy.' : 'The room has enough signal to finalize.'), 'the panel copy is not contradictory')
}

const message = 'code changed without a later passing command.'
const blocked = render([event('final', 'Room finalized.', 1000), event('check', `Final blocked: ${message}`, 2000)])
assert.equal(mergeButton(blocked).props.disabled, false, 'a live moderator block leaves the merge to the authoritative live gate')
assertPanel(blocked, false)
assert.equal(rowIcon(blocked, 'Tickets').props.name, 'check', 'the ticket row is still green on its own terms')
assert.equal(rowIcon(blocked, 'Evidence').props.name, 'check', 'the evidence row is still green on its own terms')
assert.equal(rowIcon(blocked, 'Review').props.name, 'check', 'the review row is still green on its own terms')
assert.equal(rowIcon(blocked, 'Blockers').props.name, 'clock', 'the blockers row is not ok while the moderator blocks')
assert.ok(text(blocked).includes('Final gate blocked'), 'the panel labels the moderator verdict')
assert.ok(text(blocked).includes(message), 'the panel shows the moderator message verbatim')
assert.ok(text(blocked).includes('Final blocked by the moderator'), 'the blockers row names the moderator block')
console.log('a live moderator final block closes the panel but not the merge button, whose live gate decides')

// A paired snapshot/live projection redacts the text and carries only the safe verdict bit.
const redactedBlock = render([event('final', '', 1000), { ...event('check', '', 2000), finalBlock: true }])
assertPanel(redactedBlock, false)
assert.equal(rowIcon(redactedBlock, 'Blockers').props.name, 'clock', 'a redacted block still closes the blockers row')
assert.ok(text(redactedBlock).includes('Final gate blocked'), 'a redacted block keeps the moderator verdict label')
assert.ok(text(redactedBlock).includes('Final blocked by the moderator'), 'a redacted block still names the verdict in the row')
console.log('a paired projection keeps the verdict without the redacted blocker message')

const blockedOnly = render([event('check', `Final blocked: ${message}`, 2000)])
assert.equal(mergeButton(blockedOnly).props.disabled, false, 'a block with no final event also leaves the merge to the live gate')
assertPanel(blockedOnly, false)
assert.ok(text(blockedOnly).includes(message), 'the message shows when the room never finalized')
console.log('a moderator block with no final event also holds the final readiness panel')

const stale = render([event('check', `Final blocked: ${message}`, 1000), event('final', 'Room finalized.', 2000)])
assert.equal(mergeButton(stale).props.disabled, false)
assertPanel(stale, true)
assert.equal(rowIcon(stale, 'Blockers').props.name, 'check')
assert.ok(!text(stale).includes('Final gate blocked'), 'a cleared block is not shown')
assert.ok(!text(stale).includes(message), 'a cleared block message is not shown')
console.log('a final event newer than the block restores the previous green panel')

const tieBlock = render([event('final', 'Room finalized.', 1000), event('check', `Final blocked: ${message}`, 1000)])
assertPanel(tieBlock, false)
assert.ok(text(tieBlock).includes(message), 'a block later in the array wins a same-millisecond tie')
const tieFinal = render([event('check', `Final blocked: ${message}`, 1000), event('final', 'Room finalized.', 1000)])
assertPanel(tieFinal, true)
assert.ok(!text(tieFinal).includes(message), 'a final later in the array wins a same-millisecond tie')
console.log('same-millisecond timestamps break on event order, not wall-clock')

const completedRoom = { ...room, state: 'complete' } as unknown as Room
const alreadyComplete = render([event('check', `Final blocked: ${message}`, 1000)], { room: completedRoom })
assertPanel(alreadyComplete, true)
assert.equal(mergeButton(alreadyComplete).props.disabled, false)
assert.ok(!text(alreadyComplete).includes(message), 'a stale block does not shadow a room that completed without a final event')
console.log('a complete room is not re-blocked by an older moderator block')

const notReady = render([], { leftToWork: [ticket(8)] })
assert.equal(mergeButton(notReady).props.disabled, true, 'the row predicate still gates the merge button')

const quiet = render([])
assert.equal(mergeButton(quiet).props.disabled, false, 'a room with no events keeps today behaviour')
assertPanel(quiet, true)
assert.equal(rowIcon(quiet, 'Blockers').props.name, 'check')
assert.ok(!text(quiet).includes('Final gate blocked'))
console.log('a room with no events renders the panel exactly as before')
