// Run: node electron/agent/content.test.ts
import assert from 'node:assert/strict'
import { contentText, buildUserContent } from './content.ts'
import type { Attachment } from './types/base'

const img = (name: string): Attachment => ({ id: name, kind: 'image', name, mime: 'image/png', dataUrl: `data:image/png;base64,AAA-${name}` })
const doc = (name: string, text: string): Attachment => ({ id: name, kind: 'document', name, mime: 'text/plain', text })

let passed = 0
const ok = (n: string) => { passed++; console.log(`  ✓ ${n}`) }

// contentText
assert.strictEqual(contentText('hi'), 'hi')
assert.strictEqual(contentText(null), '')
assert.strictEqual(contentText(undefined), '')
assert.strictEqual(contentText([{ type: 'text', text: 'a' }, { type: 'image_url', image_url: { url: 'x' } }, { type: 'text', text: 'b' }]), 'a\n[image]\nb')
ok('contentText flattens string, null, and content arrays')

// no attachments → plain string (back-compat)
assert.strictEqual(buildUserContent('hello', undefined), 'hello')
assert.strictEqual(buildUserContent('hello', []), 'hello')
ok('buildUserContent returns a plain string when there are no attachments')

// documents only → still a string, doc text folded in (works on every transport)
{
  const out = buildUserContent('review this', [doc('a.md', '# Title\nbody')])
  assert.strictEqual(typeof out, 'string')
  assert.ok((out as string).includes('<attached file: a.md>'))
  assert.ok((out as string).includes('# Title'))
  ok('buildUserContent folds documents into the text as labelled blocks')
}

// images → content-parts array with text + image_url blocks
{
  const out = buildUserContent('look', [img('p.png'), doc('n.txt', 'notes')])
  assert.ok(Array.isArray(out))
  const arr = out as any[]
  assert.strictEqual(arr[0].type, 'text')
  assert.ok(arr[0].text.includes('[attached image: p.png]'), 'text block names the image for non-vision seats')
  assert.ok(arr[0].text.includes('notes'), 'document text is in the text block')
  assert.strictEqual(arr[1].type, 'image_url')
  assert.strictEqual(arr[1].image_url.url, 'data:image/png;base64,AAA-p.png')
  // and the array flattens back to text with the image placeholder
  assert.ok(contentText(out).includes('[image]'))
  ok('buildUserContent builds text + image_url parts when images are attached')
}


// Accepted OCR is a plain document so every transport and history consumer sees the
// edited transcript; unaccepted and failed rows are never converted into model facts.
{
  const transcript = { id: 'ocr', kind: 'document' as const, name: 'handwriting.png', mime: 'image/png', status: 'ready' as const,
    text: '[User-reviewed OCR transcript. Recognition may contain errors.]\nMeet on Friday',
    extraction: { method: 'apple-vision' as const, state: 'accepted' as const, text: 'Meet on Friday', confidence: 1, warnings: ['Review every line.'], pageCount: 1, processedPages: 1, truncated: false } }
  const content = buildUserContent('read this', [transcript])
  assert.equal(typeof content, 'string')
  assert.match(content as string, /User-reviewed OCR transcript/)
  assert.match(content as string, /Meet on Friday/)
  assert.throws(() => buildUserContent('read this', [{ ...transcript, status: 'review', extraction: { ...transcript.extraction, state: 'review' } }]), /not ready/)
  assert.throws(() => buildUserContent('read this', [{ ...transcript, status: 'error' }]), /not ready/)
  ok('only explicitly accepted edited OCR becomes labelled model content')
}

console.log(`content tests passed (${passed})`)
