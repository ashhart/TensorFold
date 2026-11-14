import assert from 'node:assert/strict'
import { ticketReviewThreadId } from './model'

const threads: any = [{ id: 'thread-1' }, { id: 'thread-2' }]
assert.equal(ticketReviewThreadId({ threadId: 'thread-1' }, threads), 'thread-1')
assert.equal(ticketReviewThreadId({ threadId: 'missing' }, threads), null, 'a ticket whose thread was archived or deleted has no Open target')
assert.equal(ticketReviewThreadId({ threadId: null }, threads), null)
assert.equal(ticketReviewThreadId({}, threads), null)
assert.equal(ticketReviewThreadId({ threadId: 'thread-2' }, []), null, 'no loaded threads means no Open target')
console.log('ticket-review Open target passed')
