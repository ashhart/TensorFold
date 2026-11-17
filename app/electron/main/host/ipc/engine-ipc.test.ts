import assert from 'node:assert/strict'
import { engineRepo, engineRevision, engineSearchText, engineLibraryFolderId, engineLibraryScanOptions } from './engine-ipc'

// Renderer input is only ever a repo id, a revision or search text.
assert.equal(engineRepo('TensorFold/Qwen3.8-27B-MLX-4bit'), 'TensorFold/Qwen3.8-27B-MLX-4bit')
for (const bad of ['../../etc/passwd', '/abs/path', 'a/b/c', 'owner/..', '', null, 42, { repo: 'a/b' }]) {
  assert.throws(() => engineRepo(bad), /owner\/name/, String(bad))
}
assert.equal(engineRevision(undefined), null)
assert.equal(engineRevision(''), null)
assert.equal(engineRevision('93760be4f1f69842a46bc13dbdc0f19e291392a3'), '93760be4f1f69842a46bc13dbdc0f19e291392a3')
assert.throws(() => engineRevision('../main'), /branch, tag or commit/)
assert.throws(() => engineRevision(7), /branch, tag or commit/)
assert.equal(engineSearchText('  qwen  '), 'qwen')
assert.throws(() => engineSearchText(['qwen']), /must be a string/)
assert.throws(() => engineSearchText('x'.repeat(201)), /too long/)
assert.equal(engineLibraryFolderId('a'.repeat(64)), 'a'.repeat(64))
for (const bad of ['/tmp/model', '../escape', null, { path: '/tmp/model' }]) assert.throws(() => engineLibraryFolderId(bad), /saved model folder/)
assert.deepEqual(engineLibraryScanOptions(undefined), {})
assert.deepEqual(engineLibraryScanOptions({ includeHuggingFace: false }), { includeHuggingFace: false })
for (const bad of [{ path: '/tmp/model' }, { includeHuggingFace: 'yes' }, [], '/tmp/model']) assert.throws(() => engineLibraryScanOptions(bad))


console.log('engine ipc tests passed')
