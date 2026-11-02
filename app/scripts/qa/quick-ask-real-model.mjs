import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { mkdtemp, mkdir, realpath, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'
import { pipeline, TextStreamer, env } from '@huggingface/transformers'

const model = 'onnx-community/SmolLM2-135M-Instruct-ONNX'
const revision = 'b8a5c0f183b78c55955a5364f610c36668b5e681'
const cache = await realpath(await mkdtemp(path.join(os.tmpdir(), 'tensorfold-ask-cpu-model-')))
const output = process.env.TENSORFOLD_QA_OUTPUT || path.resolve('release/quick-ask-qa')
await mkdir(output, { recursive: true })
env.cacheDir = cache
const jiti = createJiti(import.meta.url)
const { generateAsk } = await jiti.import('../../electron/agent/quick-ask/generate.ts')
const { streamChat } = await jiti.import('../../electron/agent/openai.ts')
let generator, server, calls = 0
try {
  generator = await pipeline('text-generation', model, { revision, dtype: 'q4', device: 'cpu', session_options: { intraOpNumThreads: 2, interOpNumThreads: 1 } })
  server = createServer(async (req, res) => {
    try {
      let body = ''; for await (const chunk of req) body += chunk
      assert.equal(req.url, '/v1/chat/completions')
      const input = JSON.parse(body); calls++
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store' })
      const streamer = new TextStreamer(generator.tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: text => { if (!res.destroyed) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text }, finish_reason: null }] })}\n\n`) } })
      await generator(input.messages, { max_new_tokens: 32, do_sample: false, streamer })
      res.end(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
    } catch { if (!res.headersSent) res.writeHead(500); res.end() }
  })
  server.listen(0, '127.0.0.1'); await once(server, 'listening')
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`
  const provider = { id: 'cpu-qa', name: 'Isolated CPU model', baseUrl, apiKey: '', models: [model], kind: 'local', source: 'custom' }
  const binding = 'a'.repeat(64), parts = [], started = Date.now()
  for await (const step of generateAsk({ routeId: 'cpu-qa', binding, prompt: 'Answer briefly and accurately.', messages: [{ role: 'user', content: 'What is 2 + 2? Reply with the number only.' }], web: false }, new AbortController().signal, {
    resolve: async () => ({ binding, params: { model, provider, baseUrl, apiKey: '' }, assertCurrent() {} }),
    stream: streamChat, web: async () => { throw new Error('Web must remain off for this check.') },
  })) parts.push(step)
  const answer = parts.filter(part => part.type === 'content').map(part => part.text).join('')
  assert.match(answer, /\b4\b/)
  assert.equal(parts.at(-1).type, 'finish'); assert.equal(calls, 1)
  await writeFile(path.join(output, 'real-model.json'), JSON.stringify({ model, revision, runtime: 'Transformers.js 3.8.1 / ONNX Runtime, CPU q4', answer, streamedChunks: parts.filter(part => part.type === 'content').length, elapsedMs: Date.now() - started, complete: true, qualification: 'Production Ask TensorFold generation and model transport against real CPU inference. This does not qualify TensorFold engine GPU inference or model quality.' }, null, 2))
  console.log(`Real CPU model returned ${JSON.stringify(answer)} through Ask TensorFold.`)
} finally {
  if (server) { server.closeAllConnections(); server.close(); await once(server, 'close') }
  await generator?.dispose()
  await rm(cache, { recursive: true, force: true })
}
