import assert from 'node:assert/strict'
import { buildRetrievalIndex, searchRetrievalIndex } from './retrieval-index'
import { readSessionThread, searchSessionThreads } from './session-recall'
import { executeTool } from './tools'
import type { Thread } from './types'

const marker = 'synthetic-private-lavenderwidget'
const make = (id: string, incognito = false): Thread => ({ id, title: id, kind: 'task', cwd: '/synthetic', incognito, providerId: null, model: null, createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [{ role: 'user', content: `${marker} ${id}` }], timeline: [{ id: 'answer', kind: 'assistant', text: `${marker} timeline`, createdAt: 2 }], compactions: [{ id: 'compact', at: 2, trigger: 'auto', messagesFolded: 3, tokensBefore: 400, tokensAfter: 50, summaryExcerpt: `${marker} summary`, kindsDropped: ['assistant'] }] })
const privateThread = make('private', true)
const normal = make('normal')
const threads = [privateThread, normal]
const index = buildRetrievalIndex({ threads })
assert.ok(index.records.some((record) => record.sourceIds.some((id) => id.startsWith('thread:normal:'))))
assert.ok(index.records.every((record) => record.sourceIds.every((id) => !id.startsWith('thread:private:'))), 'private messages, timeline outputs and compactions never enter unified retrieval')
assert.ok(searchRetrievalIndex(index, { query: marker }).hits.every((hit) => !hit.record.id.includes('private')))
assert.deepEqual(searchSessionThreads(threads, { query: marker, includeArchived: true, includeCurrent: true }).matches.map((match) => match.threadId), ['normal'], 'portable fallback search independently excludes private history')
assert.equal(readSessionThread(threads, { threadId: privateThread.id }).thread, null, 'knowing an Incognito id cannot bypass search exclusion')
assert.equal(readSessionThread(threads, { threadId: normal.id }).thread?.id, normal.id)
const toolRead = await executeTool('session_read', { threadId: privateThread.id }, { cwd: '/synthetic', threads })
assert.equal(toolRead.ok, false)
assert.doesNotMatch(toolRead.output, /lavenderwidget/)
const toolSearch = await executeTool('session_search', { query: marker, includeCurrent: true, includeArchived: true }, { cwd: '/synthetic', threads })
assert.equal(toolSearch.ok, true)
assert.doesNotMatch(toolSearch.output, /thread:private:/)
assert.match(toolSearch.output, /thread:normal:/)
console.log('Incognito unified retrieval, portable search and direct model-facing session_read refusal pass with ordinary-history positive controls.')
