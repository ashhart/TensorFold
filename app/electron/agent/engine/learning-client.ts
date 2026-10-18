import http from 'node:http'
import type { EngineLearningFact, EngineLearningGraph } from '../types'
import type { LiveEngine } from './supervisor'

const GRAPH_BYTES = 8 * 1024 ** 2
const EVENT_BYTES = 256 * 1024
const STREAM_BYTES = 16 * 1024 ** 2
export type LessonEvent = { kind: 'fact'; id: number; text: string } | { kind: 'learning'; id: number } | { kind: 'learned'; id: number; recalled: boolean; message?: string } | { kind: 'failed'; message: string }

function endpoint(live: LiveEngine, suffix: string): URL {
  const url = new URL(live.baseUrl)
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !/^\d+$/.test(url.port) || url.pathname !== '/v1' || url.username || url.password || url.search || url.hash) throw new Error('Living Weights can only connect to the app-managed local engine.')
  url.pathname += suffix
  return url
}

export function parseLearningGraph(value: unknown): EngineLearningGraph {
  const doc = value as any
  if (!doc || !Number.isSafeInteger(doc.next) || doc.next < 0 || !Array.isArray(doc.facts) || doc.facts.length > 4096) throw new Error('The engine returned an invalid or oversized learning graph.')
  const integer = (v: any) => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0
  const numbers = (v: any, limit = 16384): number[] => Array.isArray(v) && v.length <= limit && v.every(integer) ? v : []
  const finite = (v: any) => typeof v === 'number' && Number.isFinite(v)
  const text = (v: any, max = 8192) => typeof v === 'string' && v.length <= max ? v : ''
  const facts: EngineLearningFact[] = doc.facts.map((f: any) => {
    if (!f || !integer(f.id) || !text(f.text) || !integer(f.at) || !['queued', 'learning', 'learned', 'missed', 'replaced', 'deleting', 'deleted'].includes(f.state)) throw new Error('The engine returned an invalid learning fact.')
    const out: EngineLearningFact = { id: f.id, text: f.text, source: text(f.source, 200), at: f.at, state: f.state, links: numbers(f.links, 4096) }
    for (const key of ['round', 'checks', 'since', 'step', 'of', 'deleted'] as const) if (integer(f[key])) out[key] = f[key]
    if (typeof f.recalled === 'boolean') out.recalled = f.recalled
    if (finite(f.loss)) out.loss = f.loss
    if (text(f.message)) out.message = f.message
    if (text(f.phase, 64)) out.phase = f.phase
    if (Array.isArray(f.questions) && f.questions.length <= 256) out.questions = f.questions.map((q: any) => text(q)).filter(Boolean)
    if (Array.isArray(f.neurons) && f.neurons.length <= 16384) out.neurons = f.neurons.filter((n: any) => n && integer(n.layer) && integer(n.neuron) && text(n.key, 64)).map((n: any) => ({ layer: n.layer, neuron: n.neuron, key: n.key }))
    if (Array.isArray(f.trace) && f.trace.length <= 128) out.trace = f.trace.filter((r: any) => Array.isArray(r) && r.length <= 16384 && r.every(finite))
    return out
  })
  const out: EngineLearningGraph = { next: doc.next, facts }
  const m = doc.model
  if (m && integer(m.layers) && text(m.kinds, 4096) && m.pocket && integer(m.pocket.neurons) && Array.isArray(m.shards) && m.shards.length <= 4096) {
    out.model = { layers: m.layers, kinds: m.kinds, pocket: { neurons: m.pocket.neurons, layers: numbers(m.pocket.layers), used: numbers(m.pocket.used) }, shards: m.shards.filter((s: any) => s && text(s.name, 512) && integer(s.bytes)).map((s: any) => ({ name: s.name, bytes: s.bytes, pocket: s.pocket === true, layers: numbers(s.layers) })) }
    const s = m.save
    if (s && ['saving', 'saved', 'failed'].includes(s.state) && integer(s.at)) {
      out.model.save = { state: s.state, at: s.at }
      if (finite(s.seconds) && s.seconds >= 0) out.model.save.seconds = s.seconds
      if (integer(s.bytes)) out.model.save.bytes = s.bytes
      if (Array.isArray(s.files) && s.files.length <= 4096) out.model.save.files = s.files.map((f: any) => text(f, 512)).filter(Boolean)
      if (text(s.message)) out.model.save.message = s.message
    }
  }
  return out
}

export async function readLearningGraph(live: LiveEngine, signal?: AbortSignal): Promise<EngineLearningGraph> {
  const body = await new Promise<string>((resolve, reject) => {
    const req = http.get(endpoint(live, '/slide/graph'), { agent: false, headers: { Authorization: `Bearer ${live.apiKey}` }, signal, timeout: 5000 }, (res) => {
      let bytes = 0
      const chunks: Buffer[] = []
      res.on('data', (chunk: Buffer) => { bytes += chunk.length; if (bytes > GRAPH_BYTES) { const error = new Error('The learning graph exceeded the size limit.'); reject(error); req.destroy(error) } else chunks.push(chunk) })
      res.on('end', () => res.statusCode === 200 ? resolve(Buffer.concat(chunks).toString('utf8')) : reject(new Error(`The local learning graph returned HTTP ${res.statusCode}.`)))
      res.on('error', reject)
      res.on('aborted', () => reject(new Error('The learning graph response was interrupted.')))
    })
    req.on('timeout', () => req.destroy(new Error('The learning graph request timed out.')))
    req.on('error', reject)
    req.on('socket', (socket) => socket.on('error', reject))
  })
  try { return parseLearningGraph(JSON.parse(body)) } catch (error) { if (error instanceof SyntaxError) throw new Error('The engine returned unreadable learning graph JSON.'); throw error }
}

export function streamLesson(live: LiveEngine, text: string, signal: AbortSignal, onEvent: (event: LessonEvent) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({ text, source: 'chat' })
    let complete = false
    const req = http.request(endpoint(live, '/slide/learn'), { agent: false, method: 'POST', headers: { Authorization: `Bearer ${live.apiKey}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }, signal, timeout: 10 * 60_000 }, (res) => {
      if (res.statusCode !== 200) { res.resume(); reject(Object.assign(new Error(`The local learner refused the request, HTTP ${res.statusCode}.`), { accepted: false })); return }
      if (!String(res.headers['content-type']).toLowerCase().includes('text/event-stream')) { res.resume(); reject(new Error('The learner did not return a progress stream.')); return }
      let bytes = 0
      let pending = ''
      let kind = ''
      let data: string[] = []
      let eventSize = 0
      const event = () => {
        const body = data.join('\n'); data = []; eventSize = 0
        if (body === '[DONE]') { complete = true; return }
        if (!body) { kind = ''; return }
        let doc: any
        try { doc = JSON.parse(body) } catch { throw new Error('The learner returned malformed progress JSON.') }
        const id = Number.isSafeInteger(doc?.id) && doc.id >= 0 ? doc.id : null
        if (kind === 'fact' && id != null && typeof doc.text === 'string' && doc.text.length <= 8192) onEvent({ kind, id, text: doc.text })
        else if (kind === 'learning' && id != null) onEvent({ kind, id })
        else if (kind === 'learned' && id != null && typeof doc.recalled === 'boolean') onEvent({ kind, id, recalled: doc.recalled, ...(typeof doc.message === 'string' ? { message: doc.message.slice(0, 8192) } : {}) })
        else if (kind === 'failed' && typeof doc.message === 'string') onEvent({ kind, message: doc.message.slice(0, 8192) })
        else throw new Error('The learner returned an unknown progress event.')
        kind = ''
      }
      res.setEncoding('utf8')
      res.on('data', (chunk: string) => {
        try {
          bytes += Buffer.byteLength(chunk)
          pending += chunk
          if (bytes > STREAM_BYTES || Buffer.byteLength(pending) > EVENT_BYTES) throw new Error('The learning progress stream exceeded the size limit.')
          for (;;) {
            const end = pending.indexOf('\n')
            if (end < 0) break
            const line = pending.slice(0, end).replace(/\r$/, ''); pending = pending.slice(end + 1)
            eventSize += Buffer.byteLength(line)
            if (eventSize > EVENT_BYTES) throw new Error('A learning progress event exceeded the size limit.')
            if (!line) event()
            else if (line.startsWith('event:')) kind = line.slice(6).trim()
            else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
          }
        } catch (error) { reject(error); req.destroy(error as Error) }
      })
      res.on('end', () => complete ? resolve() : reject(new Error('The learning stream ended before completion; inspect the graph before retrying.')))
      res.on('error', reject)
      res.on('aborted', () => reject(new Error('The learning connection was interrupted.')))
    })
    const deadline = setTimeout(() => req.destroy(new Error('The lesson exceeded the 30 minute limit; inspect the graph before retrying.')), 30 * 60_000)
    deadline.unref()
    req.once('close', () => clearTimeout(deadline))
    req.on('timeout', () => req.destroy(new Error('The learning progress connection timed out.')))
    req.on('error', reject)
    req.on('socket', (socket) => socket.on('error', reject))
    req.end(payload)
  })
}
