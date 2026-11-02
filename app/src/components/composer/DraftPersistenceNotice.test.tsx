import assert from 'node:assert/strict'
import { jsx, load, nodes, text } from '../reading/reading-test-harness'
import { SESSION_ONLY_DRAFT_NOTICE } from '../../lib/conversation-drafts'

const { DraftPersistenceNotice } = load('src/components/composer/DraftPersistenceNotice.tsx', {
  'react/jsx-runtime': jsx, '../../lib/icons': { Icon: 'icon' },
  '../../lib/conversation-drafts': { SESSION_ONLY_DRAFT_NOTICE },
})
let retries = 0
let tree = DraftPersistenceNotice({ error: SESSION_ONLY_DRAFT_NOTICE, sessionOnly: true, onRetry: () => retries++ })
assert.equal(tree.props.role, 'status')
assert.match(text(tree), /stays in this tab/)
assert.equal(nodes(tree).some(node => node.type === 'button'), false, 'browser capability limits do not offer a retry that cannot save')
tree = DraftPersistenceNotice({ error: 'Storage unavailable', sessionOnly: false, onRetry: () => retries++ })
assert.equal(tree.props.role, 'alert')
assert.match(text(tree), /Draft not saved/)
nodes(tree).find(node => node.type === 'button').props.onClick()
assert.equal(retries, 1, 'real persistence failures expose the working save retry')
assert.equal(DraftPersistenceNotice({ error: null, sessionOnly: false, onRetry() {} }), null)
console.log('Expected session-only draft limits and actionable persistence failures remain distinct')
