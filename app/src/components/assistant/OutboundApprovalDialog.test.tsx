import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import * as jsx from 'react/jsx-runtime'
import type { ConnectorEvent } from '../../lib/types'

const compiled = ts.transpileModule(readFileSync(path.join(__dirname, 'OutboundApprovalDialog.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', compiled)((id: string) => { const reviewed = reviewedLanguageImport(id, {}); if (reviewed.matched) return reviewed.value;
  if (id === 'react') return React
  if (id === 'react/jsx-runtime') return jsx
  if (id === '../ui') return { Button: ({ children, variant, ...props }: any) => React.createElement('button', props, children) }
  throw new Error(`Unexpected runtime dependency: ${id}`)
}, loaded, loaded.exports)
const { OutboundApprovalDialog, canReviewOutbound } = loaded.exports

const event = {
  id: 'synthetic-review', connectorId: 'whatsapp', reviewDigest: '1'.repeat(64),
  outbound: { version: 1, connectorId: 'whatsapp', accountId: 'synthetic-account', accountRevision: 'fixture', recipient: '15550000000@lid', body: '<script>not executable</script>\n' + 'Exact full message '.repeat(100), threadKey: 'whatsapp:synthetic-account:15550000000@lid' },
} as ConnectorEvent
const render = (candidate: ConnectorEvent) => renderToStaticMarkup(React.createElement(OutboundApprovalDialog, { event: candidate, busy: false, onConfirm: () => { throw new Error('Rendering must never send') }, onClose: () => {} }))
const html = render(event)
assert.ok(html.includes(event.outbound!.accountId!))
assert.ok(html.includes(event.outbound!.recipient))
assert.ok(html.includes('Exact full message '.repeat(100)), 'Approval shows the full payload rather than a truncated event summary')
assert.ok(html.includes('&lt;script&gt;not executable&lt;/script&gt;'), 'Untrusted message text remains escaped')
assert.equal(canReviewOutbound(event), true)
assert.equal(canReviewOutbound({ ...event, reviewDigest: undefined }), false)
assert.equal(canReviewOutbound({ ...event, outboundError: 'Source revoked' }), false)
assert.match(render({ ...event, outboundError: 'Source revoked' }), /disabled=""/)
