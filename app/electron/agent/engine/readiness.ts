import http from 'node:http'
import net from 'node:net'

export interface EngineModels {
  data: Array<{ id?: string; max_model_len?: unknown; context_length?: unknown }>
}

export interface EngineReadinessProbe {
  models(port: number, apiKey: string): Promise<EngineModels | null>
  context(port: number, apiKey: string, model: string, models: EngineModels): Promise<number | null>
}

export async function freeLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0, exclusive: true }, () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      server.close(() => (port ? resolve(port) : reject(new Error('No free loopback port.'))))
    })
  })
}

function engineJson(port: number, apiKey: string, method: 'GET' | 'POST', path: string, body?: unknown, timeoutMs = 2000): Promise<any | null> {
  return new Promise(resolve => {
    const payload = body === undefined ? undefined : JSON.stringify(body)
    const headers = { Authorization: `Bearer ${apiKey}`, ...(payload ? { 'Content-Type': 'application/json' } : {}) }
    const req = http.request({ host: '127.0.0.1', port, path, method, headers, timeout: timeoutMs }, res => {
      let text = ''
      res.setEncoding('utf8')
      res.on('data', chunk => { if (text.length < 65_536) text += chunk })
      res.on('end', () => {
        if (res.statusCode !== 200) return resolve(null)
        try { resolve(JSON.parse(text)) } catch { resolve(null) }
      })
      res.on('error', () => resolve(null))
      res.on('aborted', () => resolve(null))
    })
    req.on('timeout', () => req.destroy())
    req.on('error', () => resolve(null))
    req.end(payload)
  })
}

export async function probeModels(port: number, apiKey: string, timeoutMs = 2000): Promise<EngineModels | null> {
  const doc = await engineJson(port, apiKey, 'GET', '/v1/models', undefined, timeoutMs)
  return Array.isArray(doc?.data) ? doc : null
}

function positiveInt(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : null
}

export async function readContextWindow(port: number, apiKey: string, model: string, models: EngineModels | null): Promise<number | null> {
  const entry = models?.data.find(item => item?.id === model) ?? models?.data[0]
  const listed = positiveInt(entry?.max_model_len) ?? positiveInt(entry?.context_length)
  if (listed) return listed
  const tokenized = await engineJson(port, apiKey, 'POST', '/tokenize', { prompt: 'hi' })
  return positiveInt(tokenized?.max_model_len)
}

export const engineReadinessProbe: EngineReadinessProbe = { models: probeModels, context: readContextWindow }
