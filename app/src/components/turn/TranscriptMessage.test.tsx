import assert from 'node:assert/strict'
import React from 'react'
import { load, jsx } from '../reading/reading-test-harness'

const quotes: unknown[] = []
const { TranscriptMessage } = load('src/components/turn/TranscriptMessage.tsx', {
  react: { memo: React.memo, useCallback: (callback: unknown) => callback },
  'react/jsx-runtime': jsx,
  '../../lib/message-quotes': { requestMessageQuote: async (input: unknown) => { quotes.push(input) } },
  '../SentUserMessage': { SentUserMessage: 'sent-user-message' },
  '../timeline': { AssistantMessage: 'assistant-message' },
  './TurnStatsLine': { TurnStatsLine: 'turn-stats' },
})

assert.equal(TranscriptMessage.compare, null, 'Default memo comparison observes every owner, state, and item prop')
const render = TranscriptMessage.type
const item = { id: 'answer-one', kind: 'assistant', text: 'Saved answer', meta: 'Saved metadata', stats: { outputTokens: 8 } }
const props = { threadId: 'thread-one', item, busy: false, active: false, showStats: true }
let tree = render(props)
assert.equal(tree.type, 'assistant-message')
assert.equal(tree.props.text, item.text)
assert.equal(tree.props.meta, item.meta)
assert.equal(tree.props.footer.props.stats, item.stats)
const originalQuote = tree.props.onQuote
originalQuote('Exact selected passage')
assert.deepEqual(quotes.at(-1), { threadId: 'thread-one', itemId: 'answer-one', selection: 'Exact selected passage' })

tree = render({ ...props, threadId: 'thread-two', item: { ...item, id: 'answer-two', text: 'New streaming text' }, active: true, showStats: false })
assert.equal(tree.props.text, 'New streaming text')
assert.equal(tree.props.active, true)
assert.equal(tree.props.footer, undefined)
tree.props.onQuote()
assert.deepEqual(quotes.at(-1), { threadId: 'thread-two', itemId: 'answer-two' })
originalQuote()
assert.deepEqual(quotes.at(-1), { threadId: 'thread-one', itemId: 'answer-one' }, 'An old callback keeps its original owner')

tree = render({ ...props, orchestratorHostId: 'remote-owner' })
assert.equal(tree.props.onQuote, undefined, 'Remote-owned messages retain their quote boundary')
const user = { id: 'user-one', kind: 'user', text: 'Original request', quotedMessages: [{ id: 'saved-quote' }] }
tree = render({ ...props, item: user, busy: true, orchestratorHostId: 'remote-owner' })
assert.equal(tree.type, 'sent-user-message')
assert.equal(tree.props.item, user, 'Attachments, source flags, and saved quotes stay on the original item')
assert.deepEqual(tree.props.thread, { id: 'thread-one', orchestratorHostId: 'remote-owner' })
assert.equal(tree.props.busy, true, 'The live edit guard receives every busy transition')
assert.equal(tree.props.onQuote, undefined)
tree = render({ ...props, item: user, busy: false })
assert.equal(tree.props.busy, false)
tree.props.onQuote('Selected request')
assert.deepEqual(quotes.at(-1), { threadId: 'thread-one', itemId: 'user-one', selection: 'Selected request' })
assert.equal(render({ ...props, item: { id: 'tool-one', kind: 'tool' } }), null)
console.log('Memoized transcript message boundaries preserve live text, busy edit state, stats, quote identity and remote ownership')
