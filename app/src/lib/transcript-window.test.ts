import assert from 'node:assert/strict'
import { includeTranscriptRange, transcriptSegments, TRANSCRIPT_PAGE } from './transcript-window'
const total = 10000, recent = [{ start: total - TRANSCRIPT_PAGE, end: total }]
const searched = includeTranscriptRange(recent, 10, 160, total)
assert.equal(searched.reduce((sum, range) => sum + range.end - range.start, 0), 300)
assert.deepEqual(searched.at(-1), recent[0], 'searching elsewhere retains already rendered selection anchors')
const segments = transcriptSegments(total, searched)
assert.equal(segments.reduce((sum, range) => sum + range.end - range.start, 0), total)
assert.deepEqual(transcriptSegments(0, []), [])
assert.deepEqual(includeTranscriptRange([{ start: 10, end: 20 }], 15, 35, 100), [{ start: 10, end: 35 }])
assert.deepEqual(includeTranscriptRange([], -5, 100, 10), [{ start: 0, end: 10 }])
console.log('Synthetic 10k-message transcript renders 150 initial rows and retains anchors when adding a separate search page')
