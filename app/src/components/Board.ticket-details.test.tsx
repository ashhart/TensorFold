import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { fixtureText, reviewedLanguageImport } from './preferences/language-test-fixture'

const TicketDetailsButton = () => null, TicketArchiveButton = () => null
const opened: string[] = []
const ticket: any = { id: 'ticket', projectId: 'project', title: 'Full details', body: 'Untruncated text', type: 'task', priority: 'medium', effort: 'small', status: 'in_progress', tags: [], threadId: 'chat', order: 0 }
const state: any = { tickets: [ticket], rooms: [], projects: [{ id: 'project', name: 'Synthetic' }], settings: { providers: [] }, openThread: (id: string) => opened.push(id) }
const useStore = Object.assign((select: (value: any) => unknown) => select(state), { getState: () => state })
const jsx = { jsx: (type: any, props: any) => React.createElement(type, props), jsxs: (type: any, props: any) => React.createElement(type, props), Fragment: React.Fragment }
const imports: Record<string, unknown> = {
  react: { useState: (initial: unknown) => [typeof initial === 'function' ? initial() : initial, () => {}], useEffect() {}, useMemo: (make: () => unknown) => make() },
  'react/jsx-runtime': jsx,
  clsx: (...values: unknown[]) => values.filter(Boolean).join(' '),
  '../lib/preferences/useAuthoredText': { useAuthoredText: () => fixtureText },
  '../lib/preferences/language-packs': { authoredText: (_locale: string, value: string) => fixtureText(value) },
  '../lib/store': { useStore },
  '../lib/icons': { Icon: () => null },
  './ui': { Button: () => null, Menu: () => null, MenuItem: () => null },
  '../lib/policy': { allowedProviders: () => [], projectDefaultModel: () => ({}) },
  '../lib/provider-accounts': { modelDisplayName: (_provider: unknown, value: string) => value, providerModelGroups: () => [] },
  '../../shared/ticket-dependencies': { resolveTicketDependencies: () => ({ state: 'ready' }) },
  './board/TicketDetails': { TicketDetailsButton, TicketArchiveButton },
}
const loaded = { exports: {} as any }
const output = ts.transpileModule(readFileSync(path.join(__dirname, 'Board.tsx'), 'utf8') + '\nexports.Card = Card;', {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText
new Function('require', 'module', 'exports', output)((id: string) => {
  const reviewed = reviewedLanguageImport(id, imports)
  if (reviewed.matched) return reviewed.value
  assert.ok(id in imports, `Reviewed Board import ${id}`)
  return imports[id]
}, loaded, loaded.exports)

function find(node: any, type: unknown): any[] {
  if (!node || typeof node !== 'object') return []
  return [...(node.type === type ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => find(child, type))]
}
const card = loaded.exports.Card({ t: ticket, rooms: [], allTickets: [ticket], onDragStart() {}, onRun() {}, onSetMode() {} })
assert.equal(find(card, TicketDetailsButton).length, 1, 'every card needs an explicit entry point to its full editable details')
assert.equal(find(card, TicketDetailsButton)[0].props.ticket, ticket)
card.props.onClick()
assert.deepEqual(opened, ['chat'], 'existing card navigation to the worker chat remains available')
const board = loaded.exports.Board({ projectId: 'project' })
assert.equal(find(board, TicketArchiveButton).length, 1, 'the board needs a visible archive entry point')
assert.equal(find(board, TicketArchiveButton)[0].props.projectId, 'project')
console.log('Kanban cards expose details, the board exposes archives, and existing worker-chat navigation is preserved')
