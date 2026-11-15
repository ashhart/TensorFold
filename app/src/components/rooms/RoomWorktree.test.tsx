import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { fixtureText, reviewedLanguageImport } from '../preferences/language-test-fixture'

let state: any, effect: { deps: unknown[]; cleanup?: () => void } | undefined
let pendingEffect: (() => void) | undefined
const requests: Array<{ cwd: string; workspace: unknown; resolve: (branch: string | null) => void; reject: (error: Error) => void }> = []
const jsx = { jsx: React.createElement, jsxs: React.createElement, Fragment: React.Fragment }
const imports: Record<string, unknown> = {
  react: {
    useState: () => [state, (next: unknown) => { state = next }],
    useEffect: (run: () => void | (() => void), deps: unknown[]) => {
      if (!effect || deps.some((value, index) => value !== effect!.deps[index])) pendingEffect = () => {
        effect?.cleanup?.()
        const cleanup = run()
        effect = { deps, ...(cleanup ? { cleanup } : {}) }
      }
    },
  },
  'react/jsx-runtime': {
    ...jsx,
    jsx: (type: any, props: any) => React.createElement(type, props),
    jsxs: (type: any, props: any) => React.createElement(type, props),
  },
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => fixtureText },
  '../../lib/icons': { Icon: () => null },
  '../../lib/ipc': { api: { getBranch: (cwd: string, workspace: unknown) => new Promise<string | null>((resolve, reject) => {
    requests.push({ cwd, workspace, resolve, reject })
  }) } },
}
const loaded = { exports: {} as any }
const source = ts.transpileModule(readFileSync(path.join(__dirname, 'RoomWorktree.tsx'), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText
new Function('require', 'module', 'exports', source)((id: string) => {
  const reviewed = reviewedLanguageImport(id, imports)
  if (reviewed.matched) return reviewed.value
  assert.ok(id in imports, `Reviewed import ${id}`)
  return imports[id]
}, loaded, loaded.exports)

const room = (id: string, cwd: string | null, extra: Record<string, unknown> = {}) => ({ id, workspaceCwd: cwd, state: 'idle', ...extra })
function render(value: ReturnType<typeof room>) {
  const node = loaded.exports.RoomWorktree({ room: value })
  pendingEffect?.()
  pendingEffect = undefined
  return node
}
function text(node: any): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (!node || typeof node !== 'object') return ''
  return React.Children.toArray(node.props?.children).map(text).join(' ')
}
async function settle() { for (let i = 0; i < 4; i++) await Promise.resolve() }

assert.equal(render(room('new', null)), null, 'an unstarted room must not invent a worktree')
assert.equal(requests.length, 0)
const first = room('first', '/workspace/rooms/First room/code')
assert.match(text(render(first)), /First room\/code/, 'the actual worktree path is accessible while the branch loads')
assert.match(text(render(first)), /Workspace/, 'the label uses an existing translated source string')
assert.equal(requests.at(-1)?.cwd, first.workspaceCwd)
requests.at(-1)!.resolve('vontra/room-first123')
await settle()
assert.match(text(render(first)), /vontra\/room-first123/)

const second = room('second', '/workspace/rooms/Second room/code')
assert.doesNotMatch(text(render(second)), /first123/, 'navigation immediately hides the previous branch')
const secondRequest = requests.at(-1)!
const third = room('third', '/workspace/rooms/Third room/code')
render(third)
secondRequest.resolve('vontra/room-second12')
await settle()
assert.doesNotMatch(text(render(third)), /second12/, 'late results cannot cross room boundaries')
requests.at(-1)!.reject(new Error('Git unavailable'))
await settle()
assert.match(text(render(third)), /Third room\/code/)
assert.doesNotMatch(text(render(third)), /vontra\/room-/, 'a failed lookup must not fabricate a branch')

const ssh = room('ssh', '/remote/rooms/Remote room/code', { workspace: { kind: 'ssh', hostId: 'example-host', path: '/remote/project' } })
render(ssh)
assert.deepEqual(requests.at(-1)?.workspace, { kind: 'ssh', hostId: 'example-host', path: ssh.workspaceCwd })
requests.at(-1)!.resolve(null)
await settle()
assert.doesNotMatch(text(render(ssh)), /vontra\/room-/, 'a detached or unknown branch must not be guessed')

const beforeRemote = requests.length
const projected = room('projected', '/remote/rooms/Projected room/code', { orchestratorHostId: 'example-orchestrator' })
assert.match(text(render(projected)), /Projected room\/code/)
assert.equal(requests.length, beforeRemote, 'a projected remote path must never be inspected on the local host')
effect?.cleanup?.()
console.log('Room worktree identity: actual branch/path, navigation, stale replies, errors, SSH and projected rooms passed')
