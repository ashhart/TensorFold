import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const inert = () => null
const LiveMessagingSource = (_props: any) => null
const imports: Record<string, unknown> = { react: { useState: (initial: unknown) => [initial, () => {}] }, 'react/jsx-runtime': jsx, '../../../lib/ipc': { api: {} }, '../../../lib/format': { shortPath: (value: string) => value }, '../../ui': { Button: inert, Input: inert, Toggle: inert }, '../shared': { Section: inert }, './LiveMessagingSource': { LiveMessagingSource }, './MessagingSetupGuide': { MessagingSetupGuide: inert } }
function collect(node: any): any[] {
  if (!node || typeof node !== 'object') return []
  return [...(node.type === LiveMessagingSource ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap(collect)]
}
for (const brand of ['slack', 'discord']) {
  const sourceA = { id: `${brand}_a`, label: 'Example A', token: '••••••••', channels: 'C123', allowedChannels: '12345', kind: 'api_token', enabled: true, secretRotations: { token: 1 } }
  const sourceB = { ...sourceA, id: `${brand}_b`, label: 'Example B', enabled: false }
  const persisted = { [brand]: { sources: [sourceB, sourceA] } }
  const draft = { [brand]: { sources: [{ ...sourceA, token: 'new-synthetic-token', enabled: false }, sourceB] } }
  const filename = brand === 'slack' ? 'SlackConnector' : 'DiscordConnector'
  const output = ts.transpileModule(readFileSync(path.join(__dirname, `${filename}.tsx`), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  const loaded = { exports: {} as any }
  new Function('require', 'module', 'exports', output)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, `Reviewed connector adapter dependency ${name}`); return imports[name] }, loaded, loaded.exports)
  const view = loaded.exports[filename]({ ctx: { draft, persistedSettings: persisted } })
  const cards = collect(view)
  assert.equal(cards.length, 2)
  assert.equal(cards[0].props.value.id, sourceA.id)
  assert.equal(cards[0].props.value.token, 'new-synthetic-token')
  assert.equal(cards[0].props.persistedValue.id, sourceA.id, 'persisted source is resolved by account ID rather than array position')
  assert.equal(cards[0].props.persistedValue.enabled, true)
  assert.equal(cards[0].props.persistedValue.token, '••••••••')
  assert.equal(cards[1].props.persistedValue.id, sourceB.id)
  assert.equal(cards[1].props.persistedValue.enabled, false)
  const removed = loaded.exports[filename]({ ctx: { draft, persistedSettings: { [brand]: { sources: [sourceB] } } } })
  assert.equal(collect(removed)[0].props.persistedValue, null, 'a removed persisted source cannot look like an enabled saved account')
}
console.log('Slack/Discord adapter persisted Settings source binding, local secret separation and removed-source state regressions passed')
