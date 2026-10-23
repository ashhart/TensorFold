import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { randomInt } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer, request } from 'node:http'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

const app = join(dirname(fileURLToPath(import.meta.url)), '..')
const bundle = join(app, 'dist-electron/donation-node.js')
// Always build: a stale bundle would let these tests pass against earlier proxy code.
const build = spawnSync(process.execPath, ['scripts/build-donation-node.mjs'], {
  cwd: app, encoding: 'utf8', timeout: 30_000,
})
assert.ifError(build.error)
assert.equal(build.status, 0, build.stderr || build.stdout)
assert.ok(existsSync(bundle), 'the donor bundle was built')

async function listen(server) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const port = randomInt(49152, 65536)
    try {
      await new Promise((resolve, reject) => {
        const failed = (error) => { server.off('listening', ready); reject(error) }
        const ready = () => { server.off('error', failed); resolve() }
        server.once('error', failed).once('listening', ready).listen(port, '127.0.0.1')
      })
      return port
    } catch (error) {
      if (error.code !== 'EADDRINUSE') throw error
    }
  }
  throw new Error('Unable to allocate a high loopback port')
}

function close(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve())
    server.closeAllConnections()
  })
}

function call(port, path, token, body) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? undefined : JSON.stringify(body)
    const req = request({
      host: '127.0.0.1', port, path, method: data === undefined ? 'GET' : 'POST', agent: false,
      headers: {
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        ...(data ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) } : {}),
      },
    }, (res) => {
      let text = ''
      res.setEncoding('utf8').on('data', (chunk) => { text += chunk })
      res.on('error', reject).on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(text) }) } catch (error) { reject(error) }
      })
    })
    req.on('error', reject)
    req.setTimeout(3000, () => req.destroy(new Error(`Request timed out: ${path}`)))
    req.end(data)
  })
}

async function poll(check, message) {
  const deadline = Date.now() + 5000
  while (Date.now() < deadline) {
    const value = await check()
    if (value) return value
    await delay(20)
  }
  assert.fail(message)
}

const token = 'donation-test-token-0123456789'
const usage = { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }
const tokenSha = 'a'.repeat(64)
const reply = { model: 'test-model', choices: [], usage, tensorfold: { token_sha: tokenSha } }
const hits = []
let onDelayedRequest
const upstream = createServer((req, res) => {
  let data = ''
  req.setEncoding('utf8').on('data', (chunk) => { data += chunk })
  req.on('end', () => {
    hits.push({ path: req.url, body: data ? JSON.parse(data) : undefined })
    const send = () => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify(req.url === '/v1/models' ? { data: [{ id: 'test-model' }] } : reply))
    }
    if (onDelayedRequest) {
      const arrived = onDelayedRequest
      onDelayedRequest = undefined
      setTimeout(send, 400)
      arrived()
    } else send()
  })
})

let upstreamPort
async function withNode(overrides, check, invalid = false) {
  const dir = await mkdtemp(join(tmpdir(), 'donation-node-test-'))
  try {
    for (let attempt = 0; attempt < 20; attempt++) {
      const reservation = createServer()
      const port = await listen(reservation)
      await close(reservation)
      const now = Date.now()
      const card = {
        schema: 'tensorfold-donation/1', id: 'test-donor', label: 'Test donor',
        endpoint: `http://127.0.0.1:${port}`, token, model: 'test-model',
        window: { from: now - 60_000, until: now + 3_600_000 },
        caps: { maxConcurrent: 1, maxTokensPerRequest: 16, maxRequestBytes: 1024 },
        security: 'poc-insecure-lan', createdAt: now, ...overrides,
      }
      const config = join(dir, 'config.json')
      await writeFile(config, JSON.stringify({
        upstream: `http://127.0.0.1:${upstreamPort}`,
        listen: { host: invalid ? '0.0.0.0' : '127.0.0.1', port }, card,
      }))
      const child = spawn(process.execPath, [bundle, '--config', config], {
        cwd: app, stdio: ['ignore', 'pipe', 'pipe'],
      })
      let stdout = '', stderr = '', spawnError, closed = false
      child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk })
      child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk })
      child.on('error', (error) => { spawnError = error })
      const done = new Promise((resolve) => child.once('close', (code) => { closed = true; resolve(code) }))
      try {
        await poll(() => {
          assert.ifError(spawnError)
          return closed || (!invalid && stdout.includes(' listening on '))
        }, 'Daemon failed to start or exit')
        if (closed && /EADDRINUSE/.test(stderr)) continue
        if (!invalid) {
          assert.equal(closed, false, stderr)
          await poll(async () => {
            try { return (await call(port, '/health')).status === 200 } catch { return false }
          }, 'Daemon never became healthy')
        }
        await check({ port, card, stdout: () => stdout, stderr: () => stderr, done })
        return
      } finally {
        if (!closed) {
          child.kill('SIGTERM')
          const force = setTimeout(() => child.kill('SIGKILL'), 1000)
          try { await done } finally { clearTimeout(force) }
        }
      }
    }
    assert.fail('Daemon could not bind after 20 attempts')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

try {
  upstreamPort = await listen(upstream)
  await withNode({}, async ({ port, card, stdout }) => {
    const health = await call(port, '/health')
    assert.equal(health.status, 200)
    assert.equal(health.body.phase, 'open')
    console.log('PASS 1: unauthenticated health reports open phase')

    const beforeAuth = hits.length
    assert.equal((await call(port, '/v1/models')).status, 401)
    assert.equal(hits.length, beforeAuth)
    const models = await call(port, '/v1/models', token)
    assert.equal(models.status, 200)
    assert.deepEqual(models.body, { data: [{ id: 'test-model' }] })
    console.log('PASS 2: models requires the bearer token')

    const beforeCaps = hits.length
    for (const field of ['max_tokens', 'max_completion_tokens']) {
      assert.equal((await call(port, '/v1/chat/completions', token, {
        model: card.model, [field]: card.caps.maxTokensPerRequest + 1,
      })).status, 400)
      assert.equal(hits.length, beforeCaps)
    }
    console.log('PASS 4: excessive token caps rejected before upstream')

    assert.equal((await call(port, '/v1/chat/completions', token, {
      messages: [{ role: 'user', content: 'x'.repeat(card.caps.maxRequestBytes) }],
    })).status, 413)
    assert.equal(hits.length, beforeCaps)
    console.log('PASS 5: oversized request returns 413')


    let arrived = false
    onDelayedRequest = () => { arrived = true }
    const first = call(port, '/v1/chat/completions', token, { model: card.model, max_tokens: 2, stream: false })
    first.catch(() => {})
    try {
      await poll(() => arrived, 'First request never reached upstream')
      assert.equal((await call(port, '/v1/chat/completions', token, { max_tokens: 2 })).status, 429)
      assert.equal(hits.length, beforeCaps + 1)
      console.log('PASS 6: concurrent request returns 429 while upstream delays 400 ms')
      const answer = await first
      assert.equal(answer.status, 200)
      assert.deepEqual(answer.body, reply)
      const receipt = await poll(() => stdout().split('\n').filter((line) => line.startsWith('receipt '))
        .map((line) => { try { return JSON.parse(line.slice(8)) } catch { return null } })
        .find((entry) => entry?.route === '/v1/chat/completions'), 'Missing chat receipt')
      assert.equal(receipt.status, 200)
      assert.ok(Number.isFinite(receipt.ms) && receipt.ms >= 0)
      assert.deepEqual(receipt.usage, usage)
      assert.equal(receipt.tokenSha, tokenSha)
      console.log('PASS 7: chat reply and receipt preserve usage and tokenSha')
    // Caps are injected and validated, not merely capped when present.
    const beforeLimits = hits.length
    assert.equal((await call(port, '/v1/chat/completions', token, { model: card.model })).status, 200)
    assert.equal(hits[beforeLimits].body.max_tokens, card.caps.maxTokensPerRequest, 'an omitted limit becomes the card cap')
    assert.equal((await call(port, '/v1/chat/completions', token, { model: card.model, max_tokens: -1 })).status, 400)
    assert.equal((await call(port, '/v1/chat/completions', token, { model: card.model, max_tokens: '8' })).status, 400)
    assert.equal((await call(port, '/v1/chat/completions', token, { model: card.model, max_completion_tokens: card.caps.maxTokensPerRequest + 1 })).status, 400)
    assert.equal((await call(port, '/v1/chat/completions', token, { model: 'another-model' })).status, 400, 'the card model is the only one served')
    assert.equal((await call(port, '/v1/chat/completions', token, { model: card.model, n: 2 })).status, 400)
    assert.equal((await call(port, '/v1/chat/completions', token, null)).status, 400, 'a JSON null body is refused')
    const beforeResponses = hits.length
    assert.equal((await call(port, '/v1/responses', token, { model: card.model })).status, 200)
    assert.equal(hits[beforeResponses].body.max_output_tokens, card.caps.maxTokensPerRequest, 'the responses route is capped too')
    assert.deepEqual(hits.slice(beforeLimits).map((hit) => hit.path), ['/v1/chat/completions', '/v1/responses'], 'rejected limits never reach the engine')
    console.log('PASS 9: output limits are validated and injected on both routes')
    } finally { await first }
  })

  const now = Date.now()
  await withNode({ window: { from: now - 120_000, until: now - 60_000 } }, async ({ port }) => {
    const before = hits.length
    assert.equal((await call(port, '/v1/models', token)).status, 403)
    assert.equal(hits.length, before)
    const health = await call(port, '/health')
    assert.equal(health.status, 200)
    assert.equal(health.body.phase, 'after')
    console.log('PASS 3: expired window rejects models and health reports after')
  })
  await withNode({ security: 'tls', endpoint: 'https://donation.invalid' }, async ({ done, stderr }) => {
    const code = await done
    assert.ok(Number.isInteger(code) && code !== 0)
    assert.match(stderr(), /non-loopback listener/i)
    console.log('PASS 8: non-loopback TLS configuration exits non-zero')
  }, true)
  console.log('donation-node: 9 cases passed')
} finally {
  if (upstream.listening) await close(upstream)
}
