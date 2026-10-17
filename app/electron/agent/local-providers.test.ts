import assert from 'node:assert/strict'
import { contextsFromModelsPayload } from './local-providers'

// vLLM's /v1/models shape: each entry carries the window as max_model_len.
assert.deepEqual(contextsFromModelsPayload({ data: [{ id: 'qwen3.8-27b', max_model_len: 262144 }] }), { 'qwen3.8-27b': 262144 })
// The context_length spelling some compatible hosts use is heard when max_model_len is absent.
assert.deepEqual(contextsFromModelsPayload({ data: [{ id: 'm', context_length: 8192 }] }), { m: 8192 })
// ...and max_model_len wins when a host serves both names.
assert.deepEqual(contextsFromModelsPayload({ data: [{ id: 'm', max_model_len: 32768, context_length: 8192 }] }), { m: 32768 })
// Entries without a usable window or id stay out, as do junk values; a window must be a positive whole number.
assert.deepEqual(
  contextsFromModelsPayload({ data: [{ id: 'a' }, { id: 'b', max_model_len: 0 }, { id: 'c', max_model_len: -4096 }, { id: 'd', max_model_len: 1.5 }, { id: 'e', max_model_len: '4096' }] }),
  {},
)
// name and model stand in for id, bare string entries carry no window, and ids trim.
assert.deepEqual(contextsFromModelsPayload({ data: [{ name: 'n', max_model_len: 4096 }, { model: 'm2', max_model_len: 8192 }, 'plain', { max_model_len: 1 }] }), { n: 4096, m2: 8192 })
assert.deepEqual(contextsFromModelsPayload({ data: [{ id: ' spaced ', max_model_len: 4 }] }), { spaced: 4 })
// Answers that are not model lists report nothing.
assert.deepEqual(contextsFromModelsPayload(null), {})
assert.deepEqual(contextsFromModelsPayload(undefined), {})
assert.deepEqual(contextsFromModelsPayload(['x']), {})
assert.deepEqual(contextsFromModelsPayload({ error: { message: 'nope' } }), {})

console.log('local-providers: hosted model windows parse from a /v1/models payload (vLLM max_model_len)')
