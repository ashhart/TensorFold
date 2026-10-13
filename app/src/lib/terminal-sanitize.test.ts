import assert from 'node:assert/strict'
import { createTerminalSanitizer, sanitizeTerminalChunk } from './terminal-sanitize'

// OSC-8 hyperlinks are stripped wherever they appear, including split across chunks.
assert.equal(sanitizeTerminalChunk('\x1b]8;;https://evil.example\x1b\\click me\x1b]8;;\x1b\\'), 'click me')
assert.equal(sanitizeTerminalChunk('\x1b]8;;https://evil.example\x07approve\x1b]8;;\x07'), 'approve')

const sanitize = createTerminalSanitizer()
assert.equal(sanitize('ok \x1b]8;;https://evil.exa'), 'ok ')
assert.equal(sanitize('mple\x1b\\phish\x1b]8;;\x1b\\'), 'phish', 'a hyperlink split across chunks is still stripped')
assert.equal(sanitize(' plain '), ' plain ', 'carry clears after a completed sequence')

// Everything else — colors, cursor addressing, titles, plain text — passes through,
// including a title sequence split across chunks (held back, then rendered complete).
const keep = '\x1b[31mred\x1b[0m \x1b[2J\x1b[H\x1b]0;title\x07plain ✓ ü'
assert.equal(sanitizeTerminalChunk(keep), keep)

const splitTitle = createTerminalSanitizer()
assert.equal(splitTitle('a\x1b]0;tit'), 'a', 'a partial OSC opener is held back')
assert.equal(splitTitle('le\x07b'), '\x1b]0;title\x07b', 'the held-back bytes reappear once the sequence completes')
assert.equal(splitTitle('done'), 'done')

console.log('terminal sanitize tests passed')
