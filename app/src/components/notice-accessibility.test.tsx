import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import React from 'react'
import ts from 'typescript'
import { fixtureText, useAuthoredText } from './preferences/language-test-fixture'

const source = ts.createSourceFile('App.tsx', readFileSync('src/App.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const declaration = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'NoticeToast')!
const printer = ts.createPrinter()
const code = ts.transpileModule(`${printer.printNode(ts.EmitHint.Unspecified, declaration, source)}\nreturn NoticeToast;`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText
const create = (type: React.ElementType, props: { children?: React.ReactNode } & Record<string, unknown>, key: React.Key | null) => React.createElement(type, { ...props, key: key ?? undefined })
const jsx = { jsx: create, jsxs: create, Fragment: React.Fragment }
const timers: Array<{ callback: () => void; delay: number; active: boolean }> = []
const effects: Array<() => void> = []
let notice: { kind: 'info'; message: string; persistent?: true } | null = null
let dismissals = 0
const useStore = (select: (state: unknown) => unknown) => select({ notice, dismissNotice: () => { dismissals++ } })
const NoticeToast = new Function('require', 'exports', 'useAuthoredText', 'useStore', 'useEffect', 'OriginalDiagnostic', 'setTimeout', 'clearTimeout', code)(
  (id: string) => { assert.equal(id, 'react/jsx-runtime'); return jsx },
  {}, useAuthoredText, useStore, (effect: () => void) => { effects.push(effect) }, (props: { children?: React.ReactNode }) => React.createElement('span', props),
  (callback: () => void, delay: number) => { timers.push({ callback, delay, active: true }); return timers.length },
  (id: number) => { timers[id - 1].active = false },
) as () => React.ReactNode

function render(next: typeof notice) {
  notice = next
  effects.length = 0
  timers.length = 0
  const tree = NoticeToast()
  for (const effect of effects) effect()
  return tree
}
function entries(node: React.ReactNode): React.ReactElement[] {
  if (!React.isValidElement(node)) return []
  return [node, ...React.Children.toArray(node.props.children).flatMap(entries)]
}
function text(node: React.ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (!React.isValidElement(node)) return ''
  return React.Children.toArray(node.props.children).map(text).filter(Boolean).join(' ').replace(/\s+/g, ' ').trim()
}
const liveRegion = (tree: React.ReactNode) => entries(tree).find((node) => node.props.role === 'status')

const empty = render(null)
assert.ok(liveRegion(empty), 'The polite live region remains mounted before a notice arrives')
const explanation = 'Choose a folder for this conversation. Your draft and attachments stay here if you cancel.'
const persistent = render({ kind: 'info', message: explanation, persistent: true })
const persistentRegion = liveRegion(persistent)
assert.ok(persistentRegion)
assert.equal(timers.length, 0, 'The picker explanation has no expiry while the native picker has focus or after cancellation')
assert.equal(text(persistentRegion), explanation)
assert.equal(persistentRegion.props['aria-live'], 'polite')
assert.equal(persistentRegion.props['aria-atomic'], 'true')
assert.equal(dismissals, 0)
render({ kind: 'info', message: 'Settings saved' })
assert.equal(timers.length, 1)
assert.equal(timers[0].delay, 2500, 'Ordinary informational notices keep their existing lifetime')
for (const timer of timers) if (timer.active) timer.callback()
assert.equal(dismissals, 1)
console.log('Production picker notice remains in a polite live region without expiry; ordinary notices keep their timeout')
