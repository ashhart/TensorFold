import assert from 'node:assert/strict'
import { jsx, load, nodes, text } from '../reading/reading-test-harness'

const { AssistantHeader } = load('src/components/assistant/AssistantHeader.tsx', {
  'react/jsx-runtime': jsx, '../../lib/brand': { BrandIcon: 'brand', connectorBrand: () => 'source' },
  '../../lib/icons': { Icon: 'icon' }, '../ui': { Button: 'button' },
  './mappings': { sourceName: () => 'Email', timeAgo: (time: number) => `at ${time}` },
})
let refreshed = 0
const props = { snapshot: null, selectedSourceRow: null, selectedItem: null, selectedSource: 'all', openSignals: 0, urgentCount: 0, loading: true, syncingSources: false, syncingEmail: false, onRefresh: () => { refreshed++ }, onSyncAll() {}, onSyncEmail() {}, onBriefMe() {} }
const freshness = (tree: unknown) => nodes(tree).find((node) => node.props['data-testid'] === 'assistant-freshness')!
let header = AssistantHeader(props)
assert.equal(text(freshness(header)), 'Refreshing…')
assert.doesNotMatch(text(header), /Demo snapshot/, 'loading is not mistaken for demonstration data')
header = AssistantHeader({ ...props, loading: false, snapshot: { generatedAt: 1 }, refreshedAt: 50 })
assert.equal(text(freshness(header)), 'Checked at 50', 'freshness follows the last successful local inbox read, without claiming a connector sync')
header = AssistantHeader({ ...props, loading: false, refreshedAt: 50, refreshError: 'Offline' })
assert.equal(text(freshness(header)), 'Update delayed')
const refresh = nodes(header).find((node) => node.type === 'button' && text(node) === 'Refresh')!
assert.equal(refresh.props.disabled, false, 'failed reads leave a retry available')
refresh.props.onClick()
assert.equal(refreshed, 1)
header = AssistantHeader({ ...props, loading: false, refreshedAt: 50, paused: true })
assert.match(freshness(header).props.title, /resume when you return/)
console.log('Assistant header distinguishes loading, fresh data and delayed updates with a retry path')
