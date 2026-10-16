#!/usr/bin/env node
// Test double for tensorfold-native 1.0.7: the CLI text of --version, capabilities --json, models,
// info and pull, and `serve` as a tiny OpenAI-compatible server. Never loads a model or uses the GPU.
// Behaviour knobs come from FAKE_* environment variables.
import { createHash } from 'node:crypto'
import { appendFileSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync, statSync, symlinkSync, writeFileSync, writeSync, closeSync, ftruncateSync } from 'node:fs'
import http from 'node:http'
import path from 'node:path'

const env = process.env
const argv = process.argv.slice(2)
const TITLES = { qwen3_5: 'Qwen3.5-2B or Qwen3.8-27B', nemotron_h: 'Nemotron 3.5 Lightning' }
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function hub() {
  if (env.HF_HUB_CACHE) return env.HF_HUB_CACHE
  if (env.HF_HOME) return path.join(env.HF_HOME, 'hub')
  return path.join(env.HOME ?? '', '.cache', 'huggingface', 'hub')
}

function repoDir(repo) {
  const [owner, name] = repo.split('/')
  return path.join(hub(), `models--${owner}--${name}`)
}

function snapshot(repo) {
  try {
    const ref = readFileSync(path.join(repoDir(repo), 'refs', 'main'), 'utf8').trim()
    const dir = path.join(repoDir(repo), 'snapshots', ref)
    return existsSync(path.join(dir, 'config.json')) ? dir : null
  } catch { return null }
}

function record(entry) {
  if (env.FAKE_ENGINE_RECORD) appendFileSync(env.FAKE_ENGINE_RECORD, `${JSON.stringify(entry)}\n`)
}

function modelsCommand() {
  let found = 0
  let names = []
  try { names = readdirSync(hub()) } catch { /* no cache */ }
  for (const name of names.sort()) {
    if (!name.startsWith('models--')) continue
    const rest = name.slice('models--'.length)
    const sep = rest.indexOf('--')
    if (sep <= 0) continue
    const repo = `${rest.slice(0, sep)}/${rest.slice(sep + 2)}`
    const dir = snapshot(repo)
    if (!dir) continue
    const type = JSON.parse(readFileSync(path.join(dir, 'config.json'), 'utf8')).model_type
    if (!TITLES[type]) continue
    console.log(`${repo}: ${TITLES[type]} (${type})`)
    found++
  }
  if (!found) console.log('no checkpoints on disk that a registered Zig family can serve; run: tensorfold pull OWNER/NAME')
  return 0
}

function infoCommand(repo) {
  record({ kind: 'info', argument: repo })
  const dir = existsSync(path.join(repo, 'config.json')) ? repo : snapshot(repo)
  if (!dir) { console.log(`${repo} is not in the Hugging Face cache; run: tensorfold pull ${repo}`); return 1 }
  const config = JSON.parse(readFileSync(path.join(dir, 'config.json'), 'utf8'))
  if (!TITLES[config.model_type]) { console.log(`${repo}: no registered Zig family serves model_type ${config.model_type}; the 0.6 line may: tensorfold@0.6`); return 1 }
  let weights = 0
  for (const file of readdirSync(dir)) weights += statSync(path.join(dir, file)).size
  weights += Number(env.FAKE_INFO_EXTRA_BYTES ?? 0)
  const gib = 1024 ** 3
  console.log(`family: ${TITLES[config.model_type]} (${config.model_type})`)
  console.log(config.quantization ? `format: ${config.quantization.bits}-bit, group ${config.quantization.group_size}` : 'format: full precision')
  if (config.max_position_embeddings) console.log(`context: ${config.max_position_embeddings}`)
  console.log(`weights: ${(weights / gib).toFixed(2)} GiB (${weights} bytes)`)
  console.log(`memory floor: ${((weights + 2 * gib) / gib).toFixed(2)} GiB (weights plus a 2.00 GiB runtime margin)`)
  return 0
}

async function pullCommand(spec) {
  const [repo, revision = 'main'] = spec.split('@')
  record({ kind: 'pull', repo, revision, args: argv })
  if (env.FAKE_PULL_FAIL) { console.error(`${repo}: the hub's file tree is unreachable (ConnectionRefused)`); return 1 }
  const files = Number(env.FAKE_PULL_FILES ?? 3)
  const size = Number(env.FAKE_PULL_SIZE ?? 64 * 1024)
  const chunk = Number(env.FAKE_PULL_CHUNK ?? 16 * 1024)
  const sha = /^[a-f0-9]{40}$/.test(revision) ? revision : createHash('sha1').update(repo).digest('hex')
  const blobs = path.join(repoDir(repo), 'blobs')
  const snap = path.join(repoDir(repo), 'snapshots', sha)
  mkdirSync(blobs, { recursive: true })
  mkdirSync(snap, { recursive: true })
  mkdirSync(path.join(repoDir(repo), 'refs'), { recursive: true })
  console.log(`[tensorfold] downloading ${repo}@${revision} from Hugging Face`)
  const type = env.FAKE_PULL_MODEL_TYPE ?? 'qwen3_5'
  console.log(type === 'dflash' ? `${repo}: DFlash2 drafter (qwen3), for serve --drafter` : `${repo}: ${TITLES[type] ?? type} (${type})`)
  for (let i = 0; i < files; i++) {
    const blob = path.join(blobs, `blob${i}`)
    const link = path.join(snap, `model-${i}.safetensors`)
    if (existsSync(blob) && statSync(blob).size === size) {
      console.log(`  model-${i}.safetensors: cached`)
    } else {
      const fd = openSync(`${blob}.ranges`, 'a+')
      ftruncateSync(fd, size)
      for (let at = 0; at < size; at += chunk) {
        writeSync(fd, Buffer.alloc(Math.min(chunk, size - at), 1), 0, Math.min(chunk, size - at), at)
        await sleep(Number(env.FAKE_PULL_DELAY_MS ?? 0))
      }
      closeSync(fd)
      renameSync(`${blob}.ranges`, blob)
      console.log(`  model-${i}.safetensors: ${(size / 1048576).toFixed(2)} MiB in 1 pieces`)
    }
    if (!existsSync(link)) symlinkSync(blob, link)
  }
  const config = type === 'dflash' ? { model_type: 'qwen3', dflash_config: { block_size: 8 } } : { model_type: type, quantization: { bits: 4, group_size: 64 }, max_position_embeddings: 32768 }
  writeFileSync(path.join(snap, 'config.json'), JSON.stringify(config))
  writeFileSync(path.join(repoDir(repo), 'refs', 'main'), sha)
  console.log(`stored ${repo}@${sha} at ${snap}`)
  return 0
}

function flag(args, name) {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : null
}

async function serveCommand(args) {
  const model = args[0]
  const port = Number(flag(args, '--port') ?? 8080)
  const host = flag(args, '--host') ?? '127.0.0.1'
  const name = flag(args, '--name') ?? model
  const keys = (env.TENSORFOLD_API_KEY ?? '').split(',').filter(Boolean)
  const graphFile = flag(args, '--slide-graph')
  let graph = { next: 1, facts: [], model: { layers: 52, kinds: 'ME', pocket: { neurons: 64, layers: [1], used: [0] }, shards: [{ name: 'model.safetensors', bytes: 8192, pocket: true, layers: [1] }] } }
  if (graphFile && existsSync(graphFile)) graph = { ...graph, ...JSON.parse(readFileSync(graphFile, 'utf8')) }
  const saveGraph = () => { if (graphFile) writeFileSync(graphFile, JSON.stringify(graph), { mode: 0o600 }) }
  let learning = false
  record({ kind: 'serve', args, pid: process.pid, hasKey: keys.length > 0, noUpdateCheck: env.TENSORFOLD_NO_UPDATE_CHECK ?? null, secretEnv: Object.keys(env).filter((k) => /API_KEY|TOKEN|SECRET/.test(k) && k !== 'TENSORFOLD_API_KEY') })
  console.error(`[tensorfold] loading ${model}`)
  await sleep(Number(env.FAKE_SERVE_LOAD_MS ?? 50))
  if (env.FAKE_SERVE_FAIL) { console.error(`tensorfold: ${env.FAKE_SERVE_FAIL}`); process.exit(1) }
  if ((env.FAKE_CORTHEON_FAIL && args.includes('--cortheon')) || (env.FAKE_SLIDE_FAIL && args.includes('--slide'))) { console.error('tensorfold: requested mode failed to load'); process.exit(1) }
  const server = http.createServer((req, res) => {
    if (keys.length && req.headers.authorization !== `Bearer ${keys[0]}`) {
      res.writeHead(401, { 'WWW-Authenticate': 'Bearer', 'Content-Type': 'application/json' })
      res.end('{"error":{"message":"invalid api key"}}')
      return
    }
    // FAKE_CONTEXT: the served window (default 32768; 'none' = unbounded). FAKE_MODELS_NO_CONTEXT
    // leaves it out of /v1/models so only /tokenize reports it.
    const context = env.FAKE_CONTEXT === 'none' ? null : Number(env.FAKE_CONTEXT ?? 32768)
    if (env.FAKE_ENGINE_LEARNING && req.method === 'GET' && req.url === '/v1/slide/graph') {
      res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(graph)); return
    }
    if (env.FAKE_ENGINE_LEARNING && req.method === 'POST' && req.url === '/v1/slide/learn') {
      if (!args.includes('--slide')) { res.writeHead(501); res.end(); return }
      if (learning) { res.writeHead(503); res.end(); return }
      learning = true
      let body = ''
      req.on('data', (chunk) => { body += chunk })
      req.on('end', async () => {
        const input = JSON.parse(body)
        record({ kind: 'teach', text: input.text })
        const fact = { id: graph.next++, text: input.text, source: 'chat', at: Date.now(), state: 'learning', links: [] }
        graph.facts.push(fact)
        res.writeHead(200, { 'Content-Type': 'text/event-stream' })
        res.write(`event: fact\ndata: ${JSON.stringify({ id: fact.id, text: fact.text })}\n\nevent: learning\ndata: ${JSON.stringify({ id: fact.id })}\n\n`)
        await sleep(Number(env.FAKE_LEARN_DELAY_MS ?? 20))
        fact.state = 'learned'
        res.write(`event: learned\ndata: ${JSON.stringify({ id: fact.id, recalled: true })}\n\ndata: [DONE]\n\n`)
        graph.model.save = { state: 'saving', at: Date.now() }
        await sleep(Number(env.FAKE_SAVE_DELAY_MS ?? 20))
        graph.model.save.state = 'saved'
        saveGraph()
        learning = false
        res.end()
      })
      return
    }
    if (req.method === 'GET' && req.url === '/v1/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      const window = context && !env.FAKE_MODELS_NO_CONTEXT ? { context_length: context, max_model_len: context } : {}
      res.end(JSON.stringify({ object: 'list', data: [{ id: name, object: 'model', ...window }] }))
      return
    }
    if (req.method === 'POST' && req.url === '/tokenize') {
      record({ kind: 'tokenize' })
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ count: 1, max_model_len: context, tokens: [1] }))
      return
    }
    if (req.method === 'POST' && req.url === '/v1/chat/completions') {
      let body = ''
      req.on('data', (c) => { body += c })
      req.on('end', async () => {
        const request = JSON.parse(body || '{}')
        record({ kind: 'chat', model: request.model, stream: request.stream, messages: request.messages?.length ?? 0 })
        if (!request.stream) {
          res.writeHead(200, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'Fake title' }, finish_reason: 'stop' }] }))
          return
        }
        res.writeHead(200, { 'Content-Type': 'text/event-stream' })
        await sleep(Number(env.FAKE_FIRST_TOKEN_MS ?? 0))
        for (const text of ['Hello', ' from', ' the fake engine']) {
          res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text } }] })}\n\n`)
        }
        const measured = env.FAKE_REPLY_TIMINGS ? { tensorfold: { tokens_per_second: 200, seconds: 0.01, prefill_seconds: 0.05, time_to_first_token: 0.05 } } : {}
        const cacheUsage = env.FAKE_REPLY_TIMINGS ? { prompt_tokens_details: { cached_tokens: 1 } } : {}
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 3, ...cacheUsage }, ...measured })}\n\n`)
        res.end('data: [DONE]\n\n')
      })
      return
    }
    res.writeHead(404)
    res.end()
  })
  server.listen(port, host, () => console.error(`[tensorfold] serving ${name} on http://${host}:${port}/v1`))
  if (env.FAKE_SERVE_EXIT_AFTER_MS) setTimeout(() => { console.error('tensorfold: Metal device lost'); process.exit(3) }, Number(env.FAKE_SERVE_EXIT_AFTER_MS))
  const stop = () => server.close(() => process.exit(0))
  process.on('SIGTERM', env.FAKE_SERVE_IGNORE_TERM ? () => {} : stop)
}

const [command, ...rest] = argv
if (command === '--version') console.log(`tensorfold-native ${env.FAKE_ENGINE_VERSION ?? '1.0.7'}`)
else if (command === 'capabilities' && rest[0] === '--json') console.log(JSON.stringify({ schema: 1, engine: 'zig', version: '1.0.7', chip: 'Apple M5 Max', backends: ['mlx'], families: { qwen3_5: ['mlx-q4g64'], nemotron_h: ['mlx-q4g64'] }, serve: { flags: { ...(env.FAKE_ENGINE_CORTHEON ? { '--cortheon': {} } : {}), ...(env.FAKE_ENGINE_LEARNING ? { '--slide': {}, '--slide-graph': {} } : {}), '--host': {}, '--port': {}, '--drafter': {} } }, env: ['TENSORFOLD_API_KEY', 'HF_HOME', 'HF_HUB_CACHE'] }))
else if (command === 'models') process.exitCode = modelsCommand()
else if (command === 'info') process.exitCode = infoCommand(rest[0])
else if (command === 'pull') process.exitCode = await pullCommand(rest[0])
else if (command === 'serve') await serveCommand(rest)
else { console.error('usage: tensorfold-native --version | capabilities --json | models | info MODEL | pull REPO[@REVISION] | serve MODEL [flags]'); process.exitCode = 2 }
