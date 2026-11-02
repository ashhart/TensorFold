import assert from 'node:assert/strict'
import { includeTranscriptRange, restoreTranscriptRanges, transcriptSegments, TRANSCRIPT_PAGE } from './transcript-window'
import { captureTranscriptPosition, restoreTranscriptPosition, transcriptAtEnd } from './transcript-scroll'
import { createTranscriptViewMemory } from './transcript-view-memory'
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

const memory = createTranscriptViewMemory(2)
memory.set('one', { window: { ranges: searched, total }, position: { top: 120, following: false } })
memory.set('two', { position: { top: 0, following: true } })
assert.deepEqual(restoreTranscriptRanges(memory.get('one')?.window, total), searched, 'Loaded earlier pages survive leaving a thread')
const appended = restoreTranscriptRanges(memory.get('one')?.window, total + 3)
assert.deepEqual(appended[0], searched[0], 'Streaming preserves the earlier loaded page')
assert.equal(appended.at(-1)?.end, total + 3)
assert.deepEqual(restoreTranscriptRanges(memory.get('one')?.window, 10), [{ start: 0, end: 10 }], 'A reverted branch resets invalid old ranges')
assert.deepEqual(restoreTranscriptRanges({ ranges: [], total: 0 }, total), recent, 'An initially empty view does not mount the entire later transcript')
memory.set('one', { position: { top: 180, following: false } })
assert.deepEqual(memory.get('one')?.window?.ranges, searched, 'Scroll updates retain loaded ranges')
memory.set('three', { position: { top: 0, following: true } })
assert.equal(memory.get('two'), undefined, 'The renderer cache is bounded and refreshes recently written entries')
assert.equal(memory.get('one')?.position?.following, false, 'Following is stored independently for each conversation')

let anchorY = 1000
const element = {
  scrollTop: 1100, scrollHeight: 3000, clientHeight: 500,
  getBoundingClientRect: () => ({ top: 40 }),
  querySelectorAll: () => [{ dataset: { transcriptRow: 'saved-message' }, getBoundingClientRect: () => ({ top: 40 + anchorY - element.scrollTop, bottom: 440 + anchorY - element.scrollTop }) }],
} as unknown as HTMLElement
assert.equal(transcriptAtEnd(element), false)
const savedPosition = captureTranscriptPosition(element, false)
assert.deepEqual(savedPosition.anchor, { id: 'saved-message', offset: -100 })
element.scrollTop = 0; anchorY = 1200
restoreTranscriptPosition(element, savedPosition)
assert.equal(element.scrollTop, 1300, 'Restoration follows the same message after earlier content changes height')
restoreTranscriptPosition(element, { top: 9999, following: false })
assert.equal(element.scrollTop, 2500, 'Missing-anchor fallback stays within the scrollable document')
assert.equal(transcriptAtEnd(element), true)
element.scrollTop = 2440
assert.equal(transcriptAtEnd(element), false, 'A reader 60px above the bottom is no longer auto-following')
restoreTranscriptPosition(element, { top: 10, following: true })
assert.equal(element.scrollTop, 2500, 'A following conversation returns to its latest message')
console.log('Per-conversation ranges, bounded memory, anchor restoration, branch shrink and following isolation passed')
