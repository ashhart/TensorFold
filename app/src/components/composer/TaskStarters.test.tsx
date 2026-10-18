import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import { composerDraftKey } from '../../../shared/composer-drafts'
import { ConversationDrafts } from '../../lib/conversation-drafts'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const starterModule = { exports: {} as any }
const imports: Record<string, any> = { 'react/jsx-runtime': jsx, '../../lib/icons': { Icon: () => null } }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'TaskStarters.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports); return imports[id] }, starterModule, starterModule.exports)
const { TASK_STARTERS, TaskStarters } = starterModule.exports

const source = ts.createSourceFile('Composer.tsx', readFileSync(path.join(__dirname, '../Composer.tsx'), 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const composer = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'Composer')!
const printer = ts.createPrinter()
function production(names: string[], environment: Record<string, unknown>) {
  const declarations = names.map((name) => {
    const statement = composer.body!.statements.find((node) => ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === name))
    assert.ok(statement, `Production declaration ${name}`)
    return printer.printNode(ts.EmitHint.Unspecified, statement, source)
  }).join('\n')
  const code = ts.transpileModule(`${declarations}\nreturn { ${names.join(',')} };`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function(...Object.keys(environment), code)(...Object.values(environment))
}
const scope = (conversationId: string | null, incognito = false) => ({ surface: conversationId ? 'chat' as const : 'home' as const, conversationId, incognito, hostId: null })
const home = scope(null), a = scope('starter-a'), b = scope('starter-b'), privateA = scope('starter-a', true)
const drafts = new ConversationDrafts(null)
let fileClicks = 0, focuses = 0, automaticSends = 0
const activeDraftKey = { current: composerDraftKey(a) }
const fileSelectionRef: { current: ((files: File[]) => void) | null } = { current: null }
function owner(target: typeof a) {
  const current = drafts.get(target)
  return { ...current, scopeKey: composerDraftKey(target), capture: () => drafts.capture(target) }
}
function starter(target = a, overrides: Record<string, unknown> = {}) {
  const setText = (text: string) => drafts.update(target, { text })
  const chooseFiles = production(['chooseFiles'], {
    fileSelectionRef, reselectSelectionRef: { current: null }, reselectInputRef: { current: null },
    fileInputRef: { current: { click: () => fileClicks++ } },
    addFiles: (files: File[]) => { drafts.update(target, { attachments: files.map((file, index) => ({ id: `selected-${index}`, name: file.name, kind: 'document', mime: file.type, status: 'ready' })) }) },
  }).chooseFiles
  return production(['startTask'], {
    busy: false, submittingRef: { current: false }, draft: owner(target), activeDraftKey, desktopDraftPort: null,
    TASK_STARTERS, setText, taRef: { current: { focus: () => focuses++ } }, chooseFiles,
    send: () => automaticSends++, launch: () => automaticSends++, ...overrides,
  }).startTask
}
function visible(target: typeof a, overrides: Record<string, unknown> = {}) {
  const current = drafts.get(target)
  return production(['showTaskStarters'], { draft: owner(target), busy: false, submitting: false, text: current.text, attachments: current.attachments, desktopDraftPort: null,
    mode: target.conversationId ? 'thread' : 'home', thread: target.conversationId ? { timeline: [], apiMessages: [] } : undefined, ...overrides }).showTaskStarters
}
function descendants(node: any, match: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(match(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => descendants(child, match))] }
const choices: string[] = []
const rendered = TaskStarters({ onSelect: (id: string) => { choices.push(id); starter()(id) } })
const markup = renderToStaticMarkup(rendered)
assert.match(markup, /Start with a task/)
assert.match(markup, /Summarise a file/)
assert.match(markup, /Write or explain code/)
assert.match(markup, /Plan a task/)
assert.equal(descendants(rendered, (item) => item.type === 'button').length, 3)
assert.ok(visible(a)); assert.ok(visible(home))
const fileButton = descendants(rendered, (item) => item.type === 'button' && React.Children.toArray(item.props.children).includes('Summarise a file'))[0]
assert.ok(fileButton)
fileButton.props.onClick()
assert.deepEqual(choices, ['summarise'])
assert.equal(drafts.get(a).text, TASK_STARTERS[0].prompt)
assert.equal(fileClicks, 1, 'only a direct Summarise click opens the existing file picker')
assert.equal(focuses, 1)
assert.equal(automaticSends, 0, 'preparing any starter does not send or launch work')
assert.equal(visible(a), false)
assert.equal(drafts.get(b).text, '')
const selectedIntoA = fileSelectionRef.current!
activeDraftKey.current = composerDraftKey(b)
drafts.update(b, { text: 'Unfinished B' })
selectedIntoA([new File(['fixture'], 'selected-after-navigation.txt', { type: 'text/plain' })])
assert.equal(drafts.get(a).attachments[0].name, 'selected-after-navigation.txt')
assert.deepEqual(drafts.get(b).attachments, [], 'the existing picker callback stays bound to the originating conversation')
assert.equal(drafts.get(b).text, 'Unfinished B')

drafts.update(a, { text: '', attachments: [] })
starter()('plan')
assert.equal(drafts.get(a).text, '', 'an old button callback cannot seed a conversation after navigation')
activeDraftKey.current = composerDraftKey(a)
const beforeTyping = starter()
drafts.update(a, { text: 'I typed before the click settled' })
beforeTyping('code')
assert.equal(drafts.get(a).text, 'I typed before the click settled', 'the action rechecks the live draft rather than its rendered empty state')
drafts.update(a, { text: '   ' })
starter()('plan'); assert.equal(drafts.get(a).text, '   ', 'whitespace is still unfinished user input')
for (const status of ['processing', 'error', 'needs-reselection', 'review', 'ready'] as const) {
  drafts.update(a, { text: '', attachments: [{ id: 'pending', name: 'keep.txt', kind: 'document', mime: 'text/plain', status }] })
  const original = drafts.capture(a)
  starter()('code')
  assert.deepEqual(drafts.capture(a), original, `a ${status} attachment prevents replacement`)
  assert.equal(visible(a), false)
}
drafts.update(a, { text: '', attachments: [] })
for (const overrides of [{ busy: true }, { submittingRef: { current: true } }, { draft: { ...owner(a), ready: false } }, { desktopDraftPort: {}, draft: { ...owner(a), error: 'Could not load saved draft' } }]) {
  starter(a, overrides)('plan'); assert.equal(drafts.get(a).text, '')
}
for (const overrides of [{ busy: true }, { submitting: true }, { draft: { ...owner(a), ready: false } }, { desktopDraftPort: {}, draft: { ...owner(a), error: 'Could not load saved draft' } }, { thread: { timeline: [{ kind: 'user' }], apiMessages: [] } }, { thread: { timeline: [], apiMessages: [{ role: 'user' }] } }]) {
  assert.equal(visible(a, overrides), false)
}
starter()('unknown'); assert.equal(drafts.get(a).text, '')
starter()('code'); assert.equal(drafts.get(a).text, TASK_STARTERS[1].prompt)
assert.equal(fileClicks, 1, 'code and plan guidance do not request file access')
activeDraftKey.current = composerDraftKey(privateA)
starter(privateA)('plan')
assert.equal(drafts.get(privateA).text, TASK_STARTERS[2].prompt)
assert.equal(drafts.get(a).text, TASK_STARTERS[1].prompt, 'private and retained drafts are not mixed')
activeDraftKey.current = composerDraftKey(home)
starter(home)('plan'); assert.equal(drafts.get(home).text, TASK_STARTERS[2].prompt)
assert.equal(automaticSends, 0)
assert.ok(source.text.includes('{showTaskStarters && <TaskStarters onSelect={startTask} />}'), 'the tested actions are wired into the actual Composer used by Home and empty threads')
drafts.dispose()
console.log('Actual starter buttons fill only an empty ready owner draft, preserve typing/files/privacy/navigation, use the existing scoped file picker and never send automatically.')
