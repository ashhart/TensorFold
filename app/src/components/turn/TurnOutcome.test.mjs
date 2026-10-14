import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createJiti } from 'jiti'
const { TurnOutcome } = await createJiti(import.meta.url, { jsx: { runtime: 'automatic' } }).import('./TurnOutcome.tsx')
const turn = { running: false, work: [] }
assert.equal(renderToStaticMarkup(React.createElement(TurnOutcome, { turn, onShowActivity() {} })), '')
function output(verdict, extra = {}) {
  const receipt = { verdict, needsReview: !['verified', 'not_applicable'].includes(verdict), warnings: [], errors: 3, summary: 'One action was denied.', ...extra }
  return renderToStaticMarkup(React.createElement(TurnOutcome, { turn: { ...turn, work: [{ kind: 'notice', runReceipt: receipt }] }, onShowActivity() {} }))
}
assert.match(output('failed'), /Actions need review/)
assert.match(output('failed'), /3 errors were recorded/)
assert.match(output('blocked'), /One action was denied/)
assert.match(output('unverified'), /Changes need verification/)
assert.match(output('cancelled'), /Earlier actions may have completed/)
assert.match(output('not_applicable', { warnings: ['A repeat guard stopped an action.'] }), /Safety checks recorded a warning/)
assert.equal(output('verified'), '')
assert.equal(output('not_applicable'), '')
assert.match(output('failed', { warnings: undefined, errors: undefined, needsReview: false }), /The run recorded a failure/, 'older receipt fields never crash the conversation or hide the failed verdict')
assert.ok(!/automatically retry|nothing changed|rolled back/i.test(output('failed')), 'recorded failure must not imply replay or rollback')
console.log('recorded turn outcome: actual render passed')
