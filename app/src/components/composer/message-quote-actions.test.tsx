import assert from 'node:assert/strict'
import clsx from 'clsx'
import { find, harness, jsx, load, nodes } from '../reading/reading-test-harness'

const assistantHooks = harness()
const { AssistantMessage, UserBubble } = load('src/components/timeline.tsx', {
  react: assistantHooks.hooks, 'react/jsx-runtime': jsx, clsx,
  './composer/MessageQuotes': { MessageQuoteAction: 'quote-action' },
  'react-markdown': 'markdown', 'remark-gfm': {}, '../lib/markdown-citations': {}, '../lib/icons': { Icon: 'icon' }, '../lib/fileicon': {}, '../lib/highlight': {},
  './reading/math-markdown': {}, '../lib/store': { useStore: (select: any) => select({ settings: {} }) }, '../lib/activity-grouping': {}, '../lib/ipc': { api: {} },
  '../lib/voice': { speechSupported: () => false }, '../lib/diff-words': {}, './visualization': { splitVisualizationMarkdown: (text: string) => [{ type: 'markdown', text }], VisualizationCard: 'visualization' },
  '../lib/turn-blocks': { isEstimatedThroughputMeta: () => false }, './AttachmentChips': { AttachmentChips: 'attachments' }, '../lib/run-recovery': {},
})
const calls: Array<string | undefined> = []
let tree = assistantHooks.render(() => AssistantMessage({ text: 'Original assistant text', meta: 'Original diagnostic footer', onQuote: (value?: string) => calls.push(value) }))
const assistantAction = find(tree, node => node.type === 'quote-action')
assert.ok(assistantAction)
assert.equal(assistantAction.props.label, 'Quote assistant message')
const sourceWrapper = find(tree, node => node.ref === assistantAction.props.contentRef)
assert.ok(sourceWrapper)
assert.equal(nodes(sourceWrapper).some(node => node.props?.children === 'Original diagnostic footer'), false, 'Owned assistant content excludes metadata and action labels')
assistantAction.props.onQuote('Selected assistant passage'); assert.deepEqual([...calls], ['Selected assistant passage'])
tree = assistantHooks.render(() => AssistantMessage({ text: 'Original streaming answer', active: true, onQuote: () => calls.push('streaming') }))
assert.equal(nodes(tree).some(node => node.type === 'quote-action'), false, 'A changing streamed source has no quote action')
const ref = { current: null }
const user = UserBubble({ text: 'Original user text', attachments: [{ name: 'Original filename' }], contentRef: ref })
const userText = find(user, node => node.ref === ref)
assert.equal(userText.props.children, 'Original user text')
assert.equal(nodes(userText).some(node => node.type === 'attachments'), false, 'Attachment filenames cannot become a selected source passage')

const userHooks = harness()
const { SentUserMessage } = load('src/components/SentUserMessage.tsx', {
  react: userHooks.hooks, 'react/jsx-runtime': jsx, './composer/MessageQuotes': { MessageQuoteAction: 'quote-action', SentMessageQuotes: 'sent-quotes' },
  '../lib/ipc': { api: {} }, '../lib/icons': { Icon: 'icon' }, './timeline': { UserBubble: 'user-bubble' }, '../lib/store': { useStore: () => undefined },
})
const quote = { id: 'quote', sourceItemId: 'source', sourceRevision: 'a'.repeat(64), sourceText: 'Original', text: 'Edited' }
tree = userHooks.render(() => SentUserMessage({ thread: { id: 'thread-one' }, item: { id: 'user', text: 'Original sent text', quotedMessages: [quote] }, busy: false, onQuote: (value?: string) => calls.push(value) }))
const userAction = find(tree, node => node.type === 'quote-action')
assert.equal(userAction.props.label, 'Quote user message')
assert.equal(find(tree, node => node.type === 'user-bubble').props.contentRef, userAction.props.contentRef, 'Quote action owns only the exact user content node')
const sent = find(tree, node => node.type === 'sent-quotes')
assert.deepEqual(sent.props.quotes, [quote]); assert.equal(sent.props.threadId, 'thread-one')
userAction.props.onQuote('Exact user selection'); assert.equal(calls.at(-1), 'Exact user selection')
console.log('Actual user/assistant action wiring passes optional callbacks, exact owned content refs, attachment/footer exclusions, immutable sent quote snapshots and no quoting while streaming')
