import assert from 'node:assert/strict'
import * as outline from './conversation-outline'
import * as events from './search-events'
import type { SearchFilterControls as ControlsType } from './SearchFilterControls'
import { emptySearchFilterDraft } from './search-filter-draft'
import { harness, find, jsx, load, nodes, text } from '../reading/reading-test-harness'

const originals = new Map(['window', 'document'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]))
const listeners = new Map<string, Set<any>>()
Object.defineProperty(globalThis, 'window', { configurable: true, value: { addEventListener(name: string, callback: any) { if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name)!.add(callback) }, removeEventListener(name: string, callback: any) { listeners.get(name)?.delete(callback) }, dispatchEvent(event: any) { listeners.get(event.type)?.forEach(callback => callback(event)) } } })
Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {}, querySelector: () => null, activeElement: null } })
try {
  const hooks = harness()
  const { ConversationOutline } = load('src/components/search/ConversationOutline.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node },
    '../navigation/useAccessibleDialog': { useAccessibleDialog() {} }, './conversation-outline': outline, './search-events': events })
  let threadId = 'synthetic-thread'
  let timeline: any[] = Array.from({ length: 600 }, (_, index) => ({ id: `original-${index}`, kind: index % 2 ? 'assistant' : 'user', text: `Original content ${index} 日本語` }))
  const annotations = { pinned: [{ id: 'pin', itemId: 'original-20', label: 'Original pinned decision', done: false, createdAt: 1 }], notes: '' }
  const render = () => hooks.render(() => ConversationOutline({ threadId, timeline, annotations }))
  const commit = () => { let tree = render(); hooks.flush(); return render() }
  let tree = commit()
  assert.equal(nodes(tree).some(node => node.props?.role === 'dialog'), false)
  let prevented = 0
  listeners.get('keydown')?.forEach(callback => callback({ metaKey: true, shiftKey: true, key: 'O', preventDefault() { prevented++ } }))
  tree = commit(); assert.equal(prevented, 1)
  const buttons = nodes(tree).filter(node => node.props?.['data-outline-item'] !== undefined)
  assert.equal(buttons.length, outline.OUTLINE_PAGE_SIZE, 'Actual open UI renders a bounded page from 600 messages')
  const firstCallback = buttons[0].props.onClick
  firstCallback()
  const revealed = events.consumeConversationSearchTarget(threadId)
  assert.equal(revealed?.itemId, 'original-0'); assert.equal(revealed?.origin, 'outline')
  tree = commit(); assert.equal(nodes(tree).some(node => node.props?.role === 'dialog'), false)
  find(tree, node => node.props?.['aria-haspopup'] === 'dialog').props.onClick(); tree = commit()
  const filter = find(tree, node => node.type === 'select')
  filter.props.onChange({ target: { value: 'pin' } }); tree = commit()
  assert.equal(nodes(tree).filter(node => node.props?.['data-outline-item'] !== undefined).length, 1)
  assert.match(text(tree), /Original pinned decision/)
  const stale = find(tree, node => node.props?.['data-outline-item'] !== undefined)
  timeline = timeline.filter(item => item.id !== 'original-20'); tree = render()
  stale.props.onClick(); assert.equal(events.consumeConversationSearchTarget(threadId), null, 'A retained callback cannot jump to a message removed by edit/revert before effects run')
  filter.props.onChange({ target: { value: 'all' } }); tree = commit()
  const nav = find(tree, node => node.type === 'nav')
  const focused: number[] = []
  const actualButtons = [0, 1, 2].map(index => ({ focus() { focused.push(index) } }))
  ;(document as any).activeElement = actualButtons[0]
  const event = (key: string) => ({ key, preventDefault() {}, currentTarget: { querySelectorAll: () => actualButtons } })
  nav.props.onKeyDown(event('ArrowDown')); nav.props.onKeyDown(event('End')); nav.props.onKeyDown(event('ArrowUp')); nav.props.onKeyDown(event('Home'))
  assert.deepEqual(focused, [1, 2, 2, 0], 'Arrow keys wrap within outline entries; Home/End reach page boundaries')
  const latestButton = find(tree, node => node.props?.['data-outline-item'] !== undefined)
  threadId = 'different-thread'; render(); latestButton.props.onClick()
  assert.equal(events.consumeConversationSearchTarget('synthetic-thread'), null, 'A conversation switch cannot borrow the old outline')
  hooks.flush(); tree = render(); assert.equal(nodes(tree).some(node => node.props?.role === 'dialog'), false)
  hooks.cleanup(); latestButton.props.onClick(); assert.equal(events.consumeConversationSearchTarget('synthetic-thread'), null)

  const { SearchFilterControls } = load('src/components/search/SearchFilterControls.tsx', { 'react/jsx-runtime': jsx, './search-filter-draft': { emptySearchFilterDraft } }) as { SearchFilterControls: typeof ControlsType }
  let changed: any = null
  const controlTree = SearchFilterControls({ value: { ...emptySearchFilterDraft(), model: 'original-model', providerId: 'provider-a' }, onChange: value => { changed = value },
    threads: [{ providerId: 'provider-a', model: 'original-model' }, { providerId: 'provider-b', model: 'other-model' }] as any, providers: [{ id: 'provider-a', name: 'Original provider' }] })
  const selects = nodes(controlTree).filter(node => node.type === 'select')
  assert.match(text(selects[0]), /Original provider/)
  assert.doesNotMatch(text(selects[1]), /other-model/, 'Model options are constrained by the explicitly selected provider')
  selects[0].props.onChange({ target: { value: 'provider-b' } }); assert.equal(changed.model, ''); assert.equal(changed.providerId, 'provider-b')
  console.log('Actual outline and search filter components pass bounded DOM, keyboard open/movement, exact source navigation, pins, deletion/thread/unmount fences and dependent exact provider/model selection')
} finally { for (const [key, descriptor] of originals) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key) } }
