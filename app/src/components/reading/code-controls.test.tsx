import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { find, harness, jsx, load, settle, text } from './reading-test-harness'

let copied: string[] = [], result: any = { ok: true }
const clipboard = { api: { writeClipboard: async (value: string) => { copied.push(value); if (result instanceof Error) throw result; return result } } }
let highlightModule: any
const { CodeBlock: StaticBlock } = load('src/components/reading/CodeBlock.tsx', { react: React, 'react/jsx-runtime': jsx, '../../lib/ipc': clipboard, '../../lib/highlight': { highlight: (code: string, language?: string) => highlightModule.highlight(code, language) } })
highlightModule = load('src/lib/highlight.tsx', { react: React, 'react/jsx-runtime': jsx, '../components/reading/CodeBlock': { CodeBlock: StaticBlock }, '../components/reading/DiagramBlock': { DiagramBlock: 'diagram-fixture' }, '../components/reading/SafeMath': { SafeMath: 'math-fixture' }, './markdown-link': { MarkdownLink: 'a' } })
const tree = highlightModule.mdComponents.pre({ children: React.createElement('code', { className: 'language-typescript' }, 'const value = 1\n') })
const markup = renderToStaticMarkup(tree)
assert.match(markup, /Wrap/, 'Actual Markdown code blocks expose a persistent wrap control')
assert.match(markup, /Collapse code/, 'Actual Markdown code blocks expose a collapse control')
assert.match(markup, /aria-live="polite"/, 'Copy feedback is announced')
assert.match(markup, /typescript/); assert.match(markup, /max-h-\[32rem\]/)
assert.equal(highlightModule.mdComponents.a, 'a', 'Safe-link handler stays in place')
assert.equal(highlightModule.mdComponents.pre({ children: React.createElement('code', { className: 'language-mermaid' }, 'flowchart LR\nA-->B') }).type, 'diagram-fixture')
assert.equal(highlightModule.mdComponents.pre({ children: React.createElement('code', { className: 'language-math' }, 'x^2') }).type, 'math-fixture')

const hooks = harness()
const { CodeBlock } = load('src/components/reading/CodeBlock.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../../lib/ipc': clipboard, '../../lib/highlight': { highlight: (code: string) => code } })
let code = 'const value = 1\n'
const render = () => hooks.render(() => CodeBlock({ code, language: 'typescript' }))
const button = (label: string) => find(render(), (node) => node.type === 'button' && text(node) === label)
let block = render(); hooks.flush()
assert.equal(button('Wrap').props.type, 'button'); assert.equal(button('Wrap').props['aria-pressed'], false)
assert.equal(button('Collapse code').props['aria-expanded'], true)
button('Wrap').props.onClick(); assert.equal(button('Wrap').props['aria-pressed'], true)
code += 'const later = 2\n'; block = render()
assert.match(find(block, (node) => node.type === 'pre').props.className, /whitespace-pre-wrap/)
button('Collapse code').props.onClick(); block = render()
assert.ok(find(block, (node) => node.props.hidden === true)); assert.equal(button('Expand code').props['aria-expanded'], false)
button('Copy code').props.onClick(); await settle(); assert.equal(copied.at(-1), code); assert.ok(button('Copied'))
assert.match(text(find(render(), (node) => node.props.role === 'status')), /Code copied/)
result = { ok: false }; button('Copied').props.onClick(); await settle()
assert.match(text(find(render(), (node) => node.props.role === 'status')), /Copy failed/)
result = new Error('Synthetic clipboard denial.'); button('Copy code').props.onClick(); await settle()
assert.match(text(find(render(), (node) => node.props.role === 'status')), /Copy failed/)
hooks.cleanup()
console.log('Actual Markdown/code controls pass visible native keyboard controls, persistent wrap/collapse during streaming, bounded highlighting and exact clipboard success/refusal/error feedback.')
