import assert from 'node:assert/strict'
import { buildToolSchemas, executeTool, READ_ONLY_TOOLS, TOOL_SCHEMAS } from './tools.ts'
import {
  formatSessionReadResult,
  formatSessionSearchResult,
  readSessionThread,
  searchSessionThreads,
} from './session-recall'
import type { Thread } from './types'

const threads: Thread[] = [
  {
    id: 'current',
    title: 'Current task',
    kind: 'task',
    projectId: 'p1',
    cwd: '/repo',
    providerId: 'provider',
    model: 'model',
    createdAt: 10,
    updatedAt: 20,
    status: 'idle',
    apiMessages: [{ role: 'user', content: 'Current thread mentions invoices but should be excluded.' }],
    timeline: [],
  },
  {
    id: 'older',
    title: 'Invoice parser fix',
    kind: 'task',
    projectId: 'p1',
    cwd: '/repo',
    providerId: 'provider',
    model: 'model',
    createdAt: 1,
    updatedAt: 30,
    status: 'idle',
    apiMessages: [
      { role: 'user', content: 'Please repair the invoice parser.' },
      { role: 'assistant', content: 'The parser failed on multiline totals.' },
      { role: 'tool', name: 'read_file', tool_call_id: 'call_1', content: 'invoice fixture contained subtotal lines.' },
      { role: 'assistant', content: 'Fixed the multiline invoice total handling.' },
    ],
    timeline: [
      { id: 'tool_1', kind: 'tool', tool: 'run_command', title: 'npm test', text: 'invoice parser tests passed', state: 'ok', createdAt: 25 } as any,
    ],
  },
  {
    id: 'other-project',
    title: 'Invoice export in another project',
    kind: 'task',
    projectId: 'p2',
    cwd: '/other',
    providerId: 'provider',
    model: 'model',
    createdAt: 2,
    updatedAt: 40,
    status: 'idle',
    apiMessages: [{ role: 'user', content: 'Invoice export work in a different project.' }],
    timeline: [],
  },
  {
    id: 'archived',
    title: 'Archived invoice notes',
    kind: 'chat',
    projectId: 'p1',
    cwd: '/repo',
    providerId: 'provider',
    model: 'model',
    createdAt: 3,
    updatedAt: 50,
    status: 'idle',
    apiMessages: [{ role: 'user', content: 'Archived invoice note.' }],
    timeline: [],
    archivedAt: 55,
  },
]

const search = searchSessionThreads(threads, {
  query: 'invoice parser tests',
  projectId: 'p1',
  currentThreadId: 'current',
  limit: 5,
})

assert.equal(search.matches.length, 1)
assert.equal(search.matches[0].threadId, 'older')
assert.ok(search.matches[0].sourceIds.some((id) => id.startsWith('thread:older:')))
assert.doesNotMatch(formatSessionSearchResult(search), /other-project|archived|current/)
console.log('ok session search is project-scoped, bounded, and excludes current/archived threads by default')

const read = readSessionThread(threads, { threadId: 'older', maxMessages: 2 })
assert.equal(read.thread?.id, 'older')
assert.equal(read.messages.length, 3, 'bounded read keeps head marker plus tail messages')
assert.match(formatSessionReadResult(read), /Invoice parser fix/)
assert.match(formatSessionReadResult(read), /truncated 2 earlier messages/)
assert.match(formatSessionReadResult(read), /thread:older:message:3/)
console.log('ok session read returns a bounded, source-id-cited transcript window')

assert.ok(READ_ONLY_TOOLS.has('session_search'))
assert.ok(READ_ONLY_TOOLS.has('session_read'))
assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'session_search'))
assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'session_read'))
assert.ok(buildToolSchemas().some((tool: any) => tool.function?.name === 'session_search'))
assert.ok(buildToolSchemas().some((tool: any) => tool.function?.name === 'session_read'))
console.log('ok session recall schemas are exposed as read-only tools')

const toolSearch = await executeTool('session_search', {
  query: 'invoice parser',
  projectId: 'p1',
  currentThreadId: 'current',
}, { cwd: '/repo', threads })
assert.equal(toolSearch.ok, true)
assert.equal(toolSearch.title, 'Session search')
assert.match(toolSearch.output, /Invoice parser fix/)
assert.doesNotMatch(toolSearch.output, /Current task/)

const toolRead = await executeTool('session_read', { threadId: 'older', maxMessages: 1 }, { cwd: '/repo', threads })
assert.equal(toolRead.ok, true)
assert.equal(toolRead.title, 'Session read')
assert.match(toolRead.output, /Fixed the multiline invoice total handling/)
assert.match(toolRead.output, /thread:older:message:3/)
console.log('ok session recall tools execute through ToolContext threads')
