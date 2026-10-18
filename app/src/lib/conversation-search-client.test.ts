import assert from 'node:assert/strict'
import { createConversationSearchClient, searchDisplaySegments } from './conversation-search-client'
import type { ConversationSearchResult } from '../../shared/conversation-search'

async function main() {
  const pending: Array<(result: ConversationSearchResult) => void> = []
  const client = createConversationSearchClient({ conversationSearch: () => new Promise(resolve => pending.push(resolve)) })
  const result = (query: string): ConversationSearchResult => ({ query, scope: 'all', matches: [], omitted: 0, truncated: false })
  const old = client.search({ query: 'old', scope: 'all' })
  const fresh = client.search({ query: 'new', scope: 'all' })
  pending[1](result('new')); await fresh
  pending[0](result('old')); await old
  assert.equal(client.snapshot().result?.query, 'new', 'late old queries cannot replace fresh results')
  const closed = client.search({ query: 'closed', scope: 'all' }); client.clear()
  pending[2](result('closed')); await closed
  assert.equal(client.snapshot().result, null, 'a closed dialog retains no snippets')
  const malformed = client.search({ query: 'bad', scope: 'all' })
  pending[3]({ ...result('bad'), matches: [{}] } as ConversationSearchResult); await malformed
  assert.equal(client.snapshot().result, null); assert.match(client.snapshot().error, /could not finish/)
  const scoped = client.search({ query: 'inside', scope: 'thread', threadId: 'one' })
  pending[4]({ ...result('inside'), scope: 'thread', matches: [{ threadId: 'two' }] } as ConversationSearchResult); await scoped
  assert.match(client.snapshot().error, /could not finish/)
  const filterInput = { model: 'model-one' }
  const oldFilter = client.search({ query: 'same', scope: 'all', filters: filterInput })
  filterInput.model = 'mutated-input'
  assert.equal(client.snapshot().request?.filters?.model, 'model-one', 'Caller mutation cannot change a pending request filter identity')
  const newFilter = client.search({ query: 'same', scope: 'all', filters: { model: 'model-two' } })
  pending[6](result('same')); await newFilter
  pending[5](result('same')); await oldFilter
  assert.equal(client.snapshot().request?.filters?.model, 'model-two', 'Out-of-order results with identical text cannot revive older filters')
  const callsBeforeInvalid = pending.length
  await client.search({ query: 'same', scope: 'all', filters: { updatedFrom: 10, updatedBefore: 1 } })
  assert.equal(pending.length, callsBeforeInvalid, 'Invalid dates are refused before an IPC request')
  assert.match(client.snapshot().error, /search dates and filters/)
  const text = '🙂 Привет 東京 <script>'
  const segments = searchDisplaySegments(text, [{ start: 3, end: 9 }, { start: 4, end: 7 }, { start: -5, end: 2 }])
  assert.equal(segments.map(part => part.text).join(''), text)
  assert.equal(segments.filter(part => part.match).map(part => part.text).join('|'), '🙂|Привет')
  assert.equal(searchDisplaySegments('plain', [{ start: NaN, end: 8 }]).map(part => part.text).join(''), 'plain')
  console.log('Search client out-of-order/closed/scope/malformed response fences and Unicode safe highlighting passed')
}
void main().catch(error => { console.error(error); process.exitCode = 1 })
