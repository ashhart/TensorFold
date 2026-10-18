import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { authoredText } from '../../lib/preferences/language-packs'
import { jsx, load } from '../reading/reading-test-harness'
let locale = 'zh-Hans'
const original = 'Settings saved <script>provider content</script>'
const state: any = { notice: { kind: 'info', message: 'Settings saved', authored: true }, dismissNotice() {} }
const hook = { useAuthoredText: () => (source: string, params?: any) => authoredText(locale, source, params) }
const { OriginalDiagnostic } = load('src/components/OriginalDiagnostic.tsx', { 'react/jsx-runtime': jsx, '../lib/preferences/useAuthoredText': hook })
const source = readFileSync('src/App.tsx', 'utf8'), ast = ts.createSourceFile('App.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const statement = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'NoticeToast')!
const output = ts.transpileModule(`export ${statement.getText(ast)}`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const module = { exports: {} as any }
new Function('require', 'module', 'exports', 'useAuthoredText', 'useStore', 'useEffect', 'OriginalDiagnostic', output)(() => jsx, module, module.exports, hook.useAuthoredText, (select: any) => select(state), () => {}, OriginalDiagnostic)
const { NoticeToast } = module.exports
const render = () => renderToStaticMarkup(NoticeToast())
assert.ok(render().includes(authoredText(locale, 'Settings saved')))
assert.match(authoredText(locale, 'Settings saved'), /\p{Script=Han}/u)
locale = 'ja'; assert.ok(render().includes(authoredText(locale, 'Settings saved')))
assert.equal(state.notice.message, 'Settings saved', 'Language changes do not overwrite the retained authored source')
state.notice = { kind: 'error', message: original }; locale = 'zh-Hans'
const html = render()
assert.ok(html.includes(authoredText(locale, 'This action could not be completed.')))
assert.ok(html.includes(authoredText(locale, 'Original diagnostic')))
assert.ok(html.includes('Settings saved &lt;script&gt;provider content&lt;/script&gt;'))
assert.doesNotMatch(html, /<script>/)
state.notice = null; assert.equal(NoticeToast(), null)
console.log('Actual app notice view reactively translates only proven authored text and retains escaped raw provider diagnostics under a localized explanation')
