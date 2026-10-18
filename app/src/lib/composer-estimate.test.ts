import assert from 'node:assert/strict'
import { boundedComposerEstimate } from './composer-estimate'
import type { Attachment } from './types'
let reads = 0
const attachment = { name: 'fixture', get text() { reads++; return 'synthetic' } } as Attachment
const estimate = boundedComposerEstimate('draft', Array(100_000).fill(attachment))
assert.equal(estimate.partial, true); assert.equal(estimate.attachmentsCounted, 256); assert.equal(reads, 256)
assert.deepEqual(boundedComposerEstimate('', []), { tokens: 2100, partial: false, attachmentsCounted: 0 })
assert.equal(boundedComposerEstimate('abcd', []).tokens, 2101)
assert.equal(boundedComposerEstimate('x'.repeat(16_000_001), []).partial, true)
console.log('Composer estimate is explicitly approximate, bounded to256 attachments and16M characters, with no history traversal')
