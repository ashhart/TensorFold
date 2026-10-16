import assert from 'node:assert/strict'
import { runAgent } from '../loop'
import { executeTool } from '../tools'

const thread = { id: 'synthetic-private-phone-chat', phoneLocalOnly: true, kind: 'chat' }
await assert.rejects(() => runAgent({ thread, provider: { id: 'external', kind: 'cloud', baseUrl: 'https://example.test/v1' } } as any), /cannot be sent to external models/, 'external inference is refused before any model or compaction request')
const local = { id: 'tensorfold', source: 'tensorfold', kind: 'local', transport: 'chat-completions' }
await assert.rejects(() => runAgent({ thread, provider: local, toolContext: { phonePrivacyGuard: () => 'Research must be off.' } } as any), /Research must be off/, 'native web research cannot receive a phone-derived conversation')
await assert.rejects(() => runAgent({ thread, provider: local } as any), /privacy guard is unavailable/, 'private phone chats fail closed in environments without the host privacy guard')
for (const name of ['web_search', 'web_fetch']) assert.equal((await executeTool(name, { query: 'Synthetic phone note', url: 'https://example.test/' }, { cwd: '/example', phoneLocalOnly: true } as any)).ok, false, 'phone-derived content cannot be used in public web requests through a hidden tool call')
console.log('Durable phone chat routing blocks external inference, research and unguarded contexts before content leaves.')
