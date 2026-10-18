import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {
  ATTACHMENT_TEXT_MAX_BYTES,
  createAttachmentTextService,
  normalizeAttachmentTextResult,
  validateAttachmentTextRequest,
} from './attachment-text'

for (const input of [null, [], {}, { path: '/private/file', mime: 'image/png', bytes: [1] }, { url: 'https://example.invalid/image', mime: 'image/png', bytes: [1] }, { name: 'x', mime: 'text/plain', bytes: [1] }, { name: 'x', mime: 'image/png', bytes: [] }, { name: 'x', mime: 'image/png', bytes: [256] }, { name: 'x', mime: 'image/png', bytes: [-1] }, { name: 'x', mime: 'image/png', bytes: [0.5] }, { name: 'x', mime: 'image/png', bytes: 'not selected bytes' }]) assert.throws(() => validateAttachmentTextRequest(input))
assert.throws(() => validateAttachmentTextRequest({ name: 'x', mime: 'image/png', bytes: Buffer.alloc(ATTACHMENT_TEXT_MAX_BYTES + 1) }), /20 MiB/)
assert.deepEqual(validateAttachmentTextRequest({ name: 'x.png', mime: 'image/png', bytes: new Uint8Array([1, 2, 3]) }).bytes, Buffer.from([1, 2, 3]))
const result = { ok: true, text: 'meet on Friday', confidence: 0.9, warnings: [], pageCount: 1, processedPages: 1, truncated: false }
const normalized = normalizeAttachmentTextResult(result)
assert.equal(normalized.text, 'meet on Friday')
assert.match(normalized.warnings![0], /confidence score is high/)
for (const invalid of [null, { ...result, text: 'x'.repeat(199_801) }, { ...result, confidence: 2 }, { ...result, confidence: NaN }, { ...result, processedPages: 9 }, { ...result, pageCount: 0 }, { ...result, warnings: [42] }]) assert.throws(() => normalizeAttachmentTextResult(invalid))
assert.equal(normalizeAttachmentTextResult({ ok: false, error: 'unreadable image' }).error, 'unreadable image')
const unsupported = createAttachmentTextService({ helperPath: () => '/not/used', platform: 'linux' })
assert.equal((await unsupported.getCapability()).available, false)
assert.equal((await unsupported.extract({ name: 'x', mime: 'image/png', bytes: [1] })).ok, false)
unsupported.shutdown()

const root = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-attachment-text-test-'))
try {
  const fixture = path.join(root, 'helper.cjs')
  await writeFile(fixture, `
const args = process.argv.slice(2);
if (args.includes('--attachment-text-capability')) { console.log(JSON.stringify({available:true,method:'apple-vision',maxBytes:${ATTACHMENT_TEXT_MAX_BYTES},maxPages:8})); process.exit(0); }
let bytes = 0; process.stdin.on('data', chunk => bytes += chunk.length); process.stdin.on('end', () => {
  if (args.includes('hang')) { setInterval(() => {}, 1000); return; }
  if (args.includes('overflow')) { process.stdout.write('x'.repeat(1024*1024+1)); return; }
  if (args.includes('bad-json')) { console.log('not JSON'); return; }
  if (bytes === 1) { console.log(JSON.stringify({ok:false,error:'This selected file cannot be read.'})); process.exit(1); }
  console.log(JSON.stringify(${JSON.stringify(result)}));
});
`)
  const launched: Array<{ env: NodeJS.ProcessEnv; args: string[]; pid?: number }> = []
  const service = createAttachmentTextService({ helperPath: () => process.execPath, platform: 'darwin', darwinRelease: '25.0.0', spawn: (_file, args, options) => {
    const child = spawn(process.execPath, [fixture, ...args], options)
    launched.push({ env: options.env, args, pid: child.pid })
    return child
  } })
  assert.equal((await service.getCapability()).available, true)
  const files = await Promise.all([[1, 2], [3], [4, 5, 6]].map((bytes) => service.extract({ name: 'selected.png', mime: 'image/png', bytes })))
  assert.equal(files[0].ok, true)
  assert.equal(files[1].ok, false)
  // The third may wait for a free host slot in a caller; this service explicitly bounds work to two.
  if (!files[2].ok) assert.match(files[2].error ?? '', /Two files/)
  const third = await service.extract({ name: 'third.png', mime: 'image/png', bytes: [4, 5, 6] })
  assert.equal(third.ok, true)
  assert.equal(third.text, result.text)
  for (const call of launched) {
    assert.ok(!('HOME' in call.env) && !('OPENAI_API_KEY' in call.env), 'helper receives no account environment')
    assert.ok(!call.args.some((arg) => arg.includes(root)), 'renderer content paths never appear in helper arguments')
  }
  service.shutdown()
  assert.equal((await service.extract({ name: 'x', mime: 'image/png', bytes: [1, 2] })).ok, false)

  for (const mode of ['hang', 'overflow', 'bad-json']) {
    const service = createAttachmentTextService({ helperPath: () => process.execPath, platform: 'darwin', darwinRelease: '25.0.0', timeoutMs: 200, spawn: (_file, args, options) => spawn(process.execPath, [fixture, ...args, mode], options) })
    const outcome = await service.extract({ name: 'x.png', mime: 'image/png', bytes: [1, 2] })
    assert.equal(outcome.ok, false)
    assert.match(outcome.error ?? '', mode === 'hang' ? /timed out/ : mode === 'overflow' ? /output safety/ : /no usable result/)
    service.shutdown()
  }
} finally { await rm(root, { recursive: true, force: true }) }
console.log('attachment text service validates selected bytes, review output, individual failures, bounded workers, timeout and shutdown')
