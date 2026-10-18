import assert from 'node:assert/strict'
import React from 'react'
import clsx from 'clsx'
import { renderToStaticMarkup } from 'react-dom/server'
import { authoredText } from '../../lib/preferences/language-packs'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'

let locale = 'en'
const state: any = { settings: { uiPreferences: { locale } }, threads: [] }
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const { useAuthoredText } = load('src/lib/preferences/useAuthoredText.ts', {
  react: { useCallback: (callback: unknown) => callback }, '../store': { useStore }, './language-packs': { authoredText },
})
const language = { useAuthoredText }
const changeLanguage = (next: string) => { locale = next; state.settings = { ...state.settings, uiPreferences: { locale } } }
const hooks = harness(), copied: string[] = []
const original = 'Settings Cancel 你好 Привет <script>not markup</script>'
const { OriginalDiagnostic } = load('src/components/OriginalDiagnostic.tsx', {
  'react/jsx-runtime': jsx, '../lib/preferences/useAuthoredText': language,
})
const { SentUserMessage } = load('src/components/SentUserMessage.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx, '../lib/preferences/useAuthoredText': language,
  '../lib/ipc': { api: {
    writeClipboard: async (value: string) => { copied.push(value); return { ok: true } },
    prepareMessageEdit: async () => ({ mode: 'replace-chat', itemId: 'message', expectedMessageText: original, affectedItems: 0 }),
  } }, '../lib/store': { useStore }, '../lib/icons': { Icon: () => null },
  './composer/MessageQuotes': { MessageQuoteAction: () => null, SentMessageQuotes: () => null },
  './timeline': { UserBubble: ({ text: value }: any) => React.createElement('div', { 'data-user-text': true }, value) },
  './OriginalDiagnostic': { OriginalDiagnostic },
})
const props: any = { thread: { id: 'synthetic-language-chat', timeline: [] }, item: { id: 'message', text: original }, busy: false }
const render = () => hooks.render(() => SentUserMessage(props))
let tree = render()
assert.ok(find(tree, node => node.props?.['aria-label'] === 'Edit user message'))
changeLanguage('zh-Hans')
tree = render()
const chineseCopy = authoredText(locale, 'Copy')
assert.notEqual(chineseCopy, 'Copy', 'The bundled Chinese pack includes the primary action')
assert.match(renderToStaticMarkup(tree), new RegExp(chineseCopy))
assert.equal(find(tree, node => node.props?.text === original).props.text, original, 'User text is data even when it matches UI words')
const edit = find(tree, node => node.props?.['aria-label'] === authoredText(locale, 'Edit user message'))
edit.props.onClick(); await settle(); tree = render()
let input = find(tree, node => node.type === 'textarea')
const draft = '未完成の編集: Settings Cancel'
input.props.onChange({ target: { value: draft } })
changeLanguage('ru'); tree = render(); input = find(tree, node => node.type === 'textarea')
assert.equal(input.props.value, draft, 'Locale changes preserve in-progress message edits and the same hook state')
assert.equal(copied.length, 0, 'Changing a language never invokes an action')
hooks.cleanup()

changeLanguage('zh-Hans')
const diagnostic = OriginalDiagnostic({ children: 'Copy\nHTTP 503: provider original <script>value</script>' })
const diagnosticHtml = renderToStaticMarkup(diagnostic)
assert.match(diagnosticHtml, /Copy\nHTTP 503: provider original &lt;script&gt;value&lt;\/script&gt;/)
assert.doesNotMatch(diagnosticHtml, /<script>/, 'Diagnostics remain escaped data')
assert.ok(diagnostic.props['data-original-diagnostic'] !== undefined)

const approvals: string[] = []
const { ApprovalMenu } = load('src/components/composer/ApprovalMenu.tsx', {
  'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': language,
  '../../lib/preferences/language-packs': { authoredText }, '../../lib/store': { useStore },
  clsx,
  '../../lib/icons': { Icon: () => null }, '../ui': { Menu: 'menu' },
})
let menu = ApprovalMenu({ approvalMode: 'suggest', onSetApproval: (value: string) => approvals.push(value) })
let choices = menu.props.children(() => {})
const fullAccess = find(choices, node => node.type === 'button' && text(node).startsWith(authoredText(locale, 'Full access')))
fullAccess.props.onClick()
assert.deepEqual(approvals, ['auto'], 'Localized labels never replace approval policy identifiers')
changeLanguage('en'); menu = ApprovalMenu({ approvalMode: 'suggest', onSetApproval() {} }); choices = menu.props.children(() => {})
assert.ok(find(choices, node => node.type === 'button' && text(node).startsWith('Full access')), 'Static authored display getters read the new profile language')
console.log('Actual language hook and message/approval components translate authored labels, preserve unfinished edits and user text, retain policy IDs, and label escaped original diagnostics.')
