import assert from 'node:assert/strict'
import { MAX_QUOTE_TEXT, type MessageQuote } from '../../../shared/message-quotes'
import { find, harness, jsx, load, nodes, text } from '../reading/reading-test-harness'

const prior = Object.getOwnPropertyDescriptor(globalThis, 'window')
let selected: any = null
Object.defineProperty(globalThis, 'window', { configurable: true, value: { getSelection: () => selected } })
const ownNode = {}, foreignNode = {}, rootNode = {}
const element: any = { contains: (node: unknown) => node === ownNode || node === rootNode }
const selection = (value: string, anchor = ownNode, focus = ownNode, ancestor = rootNode) => ({ isCollapsed: false, rangeCount: 1, anchorNode: anchor, focusNode: focus, getRangeAt: () => ({ commonAncestorContainer: ancestor }), toString: () => value })
try {
  const hooks = harness(), revealed: any[] = [], calls: (string | undefined)[] = []
  const module = load('src/components/composer/MessageQuotes.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx,
    '../../../shared/message-quotes': { MAX_QUOTE_TEXT }, '../search/search-events': { revealConversationItem: (...args: any[]) => revealed.push(args) } })
  selected = selection('Owned exact passage')
  assert.equal(module.ownedMessageSelection(element), 'Owned exact passage')
  for (const value of [selection('Foreign selection', foreignNode), selection('Cross-message range', ownNode, foreignNode), selection('Outside ancestor', ownNode, ownNode, foreignNode), { ...selection('Multiple ranges'), rangeCount: 2 }, { ...selection('Collapsed'), isCollapsed: true }, selection('  ')]) {
    selected = value; assert.equal(module.ownedMessageSelection(element), undefined, 'Only a nonempty range wholly inside the owned content can become a selection quote')
  }
  selected = selection('x'.repeat(MAX_QUOTE_TEXT + 1000))
  assert.equal(module.ownedMessageSelection(element).length, MAX_QUOTE_TEXT + 1, 'Oversized selections stay a bounded explicit refusal rather than silently becoming an accepted truncated quotation')
  const action = () => hooks.render(() => module.MessageQuoteAction({ contentRef: { current: element }, onQuote: (value?: string) => calls.push(value), label: 'Quote user message' }))
  selected = selection('Pointer-selected original')
  let button = action(); button.props.onMouseDown({ button: 0 }); selected = null; button.props.onClick()
  assert.deepEqual(calls, ['Pointer-selected original'], 'Pointer focus may clear browser selection after mousedown; the owned snapshot survives until click')
  selected = selection('Foreign selection', foreignNode); button = action(); button.props.onClick(); assert.equal(calls.at(-1), undefined, 'An unrelated selection requests the actual source message, not foreign text')
  selected = selection('Keyboard selection'); button = action(); button.props.onClick(); assert.equal(calls.at(-1), 'Keyboard selection')

  const draftHooks = harness(), changes: any[] = [], removes: any[] = [], editingEvents: boolean[] = []
  const reportEditing = (value: boolean) => editingEvents.push(value)
  let threadId = 'chat-one', disabled = false
  const quote: MessageQuote = { id: 'quote-one', sourceItemId: 'source-one', sourceRevision: 'a'.repeat(64), sourceText: 'Original source text', text: 'Original source text' }
  let quotes: MessageQuote[] = [quote]
  const draftModule = load('src/components/composer/MessageQuotes.tsx', { react: draftHooks.hooks, 'react/jsx-runtime': jsx,
    '../../../shared/message-quotes': { MAX_QUOTE_TEXT }, '../search/search-events': { revealConversationItem: (...args: any[]) => revealed.push(args) } })
  const build = () => draftModule.DraftMessageQuotes({ quotes, threadId, disabled, onChange: (...args: any[]) => changes.push(args), onRemove: (id: string) => removes.push(id), onEditingChange: reportEditing })
  const render = () => { draftHooks.render(build); draftHooks.flush(); return draftHooks.render(build) }
  const getButton = (tree: any, label: string) => find(tree, node => node.type === 'button' && text(node) === label)
  let tree = render(); assert.match(tree.props.className, /max-h-64/); assert.match(tree.props.className, /overflow-y-auto/, 'Multiple quote cards scroll inside a bounded region instead of displacing the composer'); getButton(tree, 'Open source message').props.onClick(); assert.deepEqual(revealed.at(-1), [{ threadId: 'chat-one', itemId: 'source-one' }, 'quote'])
  getButton(tree, 'Edit quoted passage').props.onClick(); tree = render(); assert.equal(editingEvents.at(-1), true)
  find(tree, node => node.type === 'textarea').props.onChange({ target: { value: 'Cancelled text' } }); tree = render()
  getButton(tree, 'Cancel quotation edit').props.onClick(); tree = render(); assert.equal(changes.length, 0); assert.equal(quote.text, quote.sourceText); assert.equal(editingEvents.at(-1), false)
  getButton(tree, 'Edit quoted passage').props.onClick(); tree = render(); assert.equal(editingEvents.at(-1), true)
  find(tree, node => node.type === 'textarea').props.onChange({ target: { value: 'User-edited quote 日本語' } }); tree = render()
  assert.equal(changes.length, 0, 'Typing is a local preview until explicit save')
  getButton(tree, 'Save quotation').props.onClick(); assert.deepEqual(changes, [['quote-one', 'User-edited quote 日本語']]); assert.equal(editingEvents.at(-1), false); assert.equal(quote.sourceText, 'Original source text')
  quotes = [{ ...quote, text: 'User-edited quote 日本語' }]; tree = render(); assert.match(text(tree), /Edited quotation/)
  getButton(tree, 'Edit quoted passage').props.onClick(); tree = render(); assert.equal(editingEvents.at(-1), true)
  find(tree, node => node.type === 'textarea').props.onChange({ target: { value: ' ' } }); tree = render()
  assert.equal(getButton(tree, 'Save quotation').props.disabled, true); getButton(tree, 'Save quotation').props.onClick(); assert.equal(changes.length, 1)
  getButton(tree, 'Cancel quotation edit').props.onClick(); tree = render()
  const retainedRemove = getButton(tree, 'Remove quoted passage').props.onClick
  const retainedSource = getButton(tree, 'Open source message').props.onClick
  disabled = true; tree = render(); retainedRemove(); assert.equal(removes.length, 0)
  disabled = false; threadId = 'chat-two'; tree = render(); retainedRemove(); const before = revealed.length; retainedSource(); assert.equal(revealed.length, before, 'A retained card callback cannot jump after a chat change')
  assert.equal(removes.length, 0)
  getButton(tree, 'Remove quoted passage').props.onClick(); assert.deepEqual(removes, ['quote-one'])
  const sent = draftHooks.render(() => draftModule.SentMessageQuotes({ quotes, threadId }))
  assert.equal(nodes(sent).some(node => node.type === 'textarea'), false)
  assert.equal(nodes(sent).filter(node => node.type === 'button').length, 1, 'A sent quote has source navigation only, no edit/remove action')
  assert.match(text(sent), /Edited quotation/)
  // External removal and owner changes release the blocking state without committing unsaved text.
  tree = render(); getButton(tree, 'Edit quoted passage').props.onClick(); tree = render(); quotes = []; render(); assert.equal(editingEvents.at(-1), false)
  quotes = [{ ...quote }]; tree = render(); getButton(tree, 'Edit quoted passage').props.onClick(); render(); threadId = 'chat-three'; render(); assert.equal(editingEvents.at(-1), false)
  tree = render(); getButton(tree, 'Edit quoted passage').props.onClick(); render(); draftHooks.cleanup(); assert.equal(editingEvents.at(-1), false)
  // Two keyed composers may overlap during commit. An old unmount cannot unlock a newer editor.
  const oldHooks = harness(), newHooks = harness(); let activeHooks = oldHooks
  const lifecycle: boolean[] = [], callback = (value: boolean) => lifecycle.push(value)
  const sharedModule = load('src/components/composer/MessageQuotes.tsx', { react: { useState: (...args: any[]) => activeHooks.hooks.useState(...args), useRef: (...args: any[]) => activeHooks.hooks.useRef(...args), useEffect: (...args: any[]) => activeHooks.hooks.useEffect(...args) }, 'react/jsx-runtime': jsx,
    '../../../shared/message-quotes': { MAX_QUOTE_TEXT }, '../search/search-events': { revealConversationItem() {} } })
  const ownerTree = (owner: ReturnType<typeof harness>, id: string) => { activeHooks = owner; const buildOwner = () => sharedModule.DraftMessageQuotes({ quotes: [quote], threadId: id, onChange() {}, onEditingChange: callback }); owner.render(buildOwner); owner.flush(); return owner.render(buildOwner) }
  let oldTree = ownerTree(oldHooks, 'old-owner'); getButton(oldTree, 'Edit quoted passage').props.onClick(); ownerTree(oldHooks, 'old-owner')
  let newTree = ownerTree(newHooks, 'new-owner'); getButton(newTree, 'Edit quoted passage').props.onClick(); ownerTree(newHooks, 'new-owner')
  const countBeforeOldCleanup = lifecycle.length; oldHooks.cleanup(); assert.equal(lifecycle.length, countBeforeOldCleanup, 'Old editor cleanup cannot clear a newer editor owned by the same callback')
  newHooks.cleanup(); assert.equal(lifecycle.at(-1), false)
  console.log('Quote UI passes owned text selection, pointer/keyboard capture, bounded refusal, explicit save/cancel, original provenance, blank refusal, immutable sent cards and disabled/chat stale callback fences')
} finally { if (prior) Object.defineProperty(globalThis, 'window', prior); else Reflect.deleteProperty(globalThis, 'window') }
