import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { PROVIDER_SSE_FRAME_MAX_CHARS, streamChat } from './openai'

let responseBody = Buffer.alloc(0)
let fragment = false
let keepOpen = false
const server = createServer(async (request, response) => {
  request.resume()
  response.writeHead(200, { 'Content-Type': 'text/event-stream' })
  if (fragment) {
    for (const byte of responseBody) {
      if (response.destroyed) return
      response.write(Buffer.from([byte]))
      await new Promise<void>((resolve) => setImmediate(resolve))
    }
  } else response.write(responseBody)
  if (!keepOpen) response.end()
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`

async function collect(body: string, options: { fragment?: boolean; keepOpen?: boolean } = {}) {
  responseBody = Buffer.from(body)
  fragment = options.fragment ?? false
  keepOpen = options.keepOpen ?? false
  const chunks: any[] = []
  for await (const chunk of streamChat({ baseUrl, apiKey: '', model: 'fixture', messages: [{ role: 'user', content: 'hello' }] })) chunks.push(chunk)
  return chunks
}
function event(value: unknown, lineEnding = '\n') {
  return JSON.stringify(value, null, 2).split('\n').map((line) => `data: ${line}`).join(lineEnding) + lineEnding + lineEnding
}
const text = (chunks: any[]) => chunks.filter((chunk) => chunk.type === 'content').map((chunk) => chunk.text).join('')
const failures: string[] = []
async function check(name: string, run: () => Promise<void>) {
  try { await run(); console.log(`PASS ${name}`) } catch (error) { failures.push(`${name}: ${(error as Error).message}`) }
}
try {
  await check('one event with two data lines retains content', async () => {
    const body = 'data: {"choices":[{"delta":\ndata: {"content":"ok"}}]}\n\n'
    assert.equal(text(await collect(body)), 'ok')
  })
  await check('multiline provider error fails instead of a blank success', async () => {
    await assert.rejects(() => collect(event({ error: { message: 'multiline failure' } })), /Provider error: multiline failure/)
  })
  await check('multiline tool and usage events preserve the complete call and actual counts', async () => {
    const tool = { id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"README.md"}' } }
    const chunks = await collect(event({ choices: [{ delta: { tool_calls: [{ index: 0, ...tool }] }, finish_reason: 'tool_calls' }] })
      + event({ choices: [], usage: { prompt_tokens: 12, completion_tokens: 7 } }) + 'data: [DONE]\n\n')
    assert.deepEqual(chunks.find((chunk) => chunk.type === 'tool_calls')?.toolCalls, [tool])
    assert.equal(chunks.find((chunk) => chunk.type === 'usage')?.completionTokens, 7)
    assert.equal(chunks.filter((chunk) => chunk.type === 'finish').length, 1)
  })
  for (const ending of ['\n', '\r\n', '\r']) {
    await check(`fragmented UTF-8 with ${JSON.stringify(ending)} line endings`, async () => {
      const chunks = await collect(`: heartbeat${ending}event: message${ending}id: ignored${ending}`
        + event({ choices: [{ delta: { content: 'café 😀' }, finish_reason: 'stop' }] }, ending)
        + `data: [DONE]${ending}${ending}`, { fragment: true })
      assert.equal(text(chunks), 'café 😀')
      assert.equal(chunks.filter((chunk) => chunk.type === 'finish').length, 1)
    })
  }
  await check('DONE ends the read even while the server keeps the body open', async () => {
    const chunks = await collect(event({ choices: [{ delta: { content: 'done' } }] }) + 'data: [DONE]\n\n', { keepOpen: true })
    assert.equal(text(chunks), 'done')
    assert.equal(chunks.filter((chunk) => chunk.type === 'finish').length, 1)
  })
  await check('EOF compatibility retains a final data line without a delimiter', async () => {
    assert.equal(text(await collect('data: {"choices":[{"delta":{"content":"tail"},"finish_reason":"stop"}]}')), 'tail')
  })
  await check('frame limit bounds the entire multiline event', async () => {
    const line = `data: ${'x'.repeat(128 * 1024)}\n`
    await assert.rejects(() => collect(line.repeat(Math.ceil(PROVIDER_SSE_FRAME_MAX_CHARS / (128 * 1024)) + 1) + '\n'), /frame exceeded the safety limit/)
  })
  assert.equal(failures.length, 0, failures.join('\n'))
  console.log('Real HTTP SSE framing tests passed')
} finally {
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
}
