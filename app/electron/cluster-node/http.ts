import type { IncomingMessage, ServerResponse } from 'node:http'
import { NodeHttpError } from './config'

export function json(response: ServerResponse, status: number, value: unknown): void {
  if (response.destroyed) return
  const bytes = Buffer.from(JSON.stringify(value))
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'content-length': bytes.length, 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer' })
  response.end(bytes)
}
export async function readJson(request: IncomingMessage, maximum: number): Promise<Record<string, any>> {
  if (!/^application\/json(?:\s*;.*)?$/i.test(request.headers['content-type'] ?? '')) throw new NodeHttpError(415, 'Use a JSON request body.')
  if (request.headers['content-encoding']) throw new NodeHttpError(415, 'Compressed management and inference bodies are not accepted.')
  const length = request.headers['content-length']
  if (length !== undefined && (!/^\d+$/.test(length) || Number(length) > maximum)) throw new NodeHttpError(413, 'The request exceeds its byte limit.')
  const chunks: Buffer[] = []; let bytes = 0
  for await (const chunk of request) { bytes += chunk.length; if (bytes > maximum) throw new NodeHttpError(413, 'The request exceeds its byte limit.'); chunks.push(Buffer.from(chunk)) }
  let value: unknown
  try { value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks, bytes))) } catch { throw new NodeHttpError(400, 'The JSON body is malformed or incomplete.') }
  let nodes = 0
  const bounded = (value: any, depth = 0) => {
    if (++nodes > 50_000 || depth > 24) throw new NodeHttpError(413, 'The JSON structure exceeds its depth or item limit.')
    if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) { if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new NodeHttpError(400, 'Unsafe JSON keys are not accepted.'); bounded(child, depth + 1) }
  }
  bounded(value)
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new NodeHttpError(400, 'The JSON body must be an object.')
  return value as Record<string, any>
}
export async function writeChunk(response: ServerResponse, chunk: Uint8Array, signal: AbortSignal): Promise<void> {
  if (signal.aborted || response.destroyed) throw new Error('The inference client disconnected.')
  if (!response.write(chunk)) await new Promise<void>((resolve, reject) => {
    const clear = () => { response.removeListener('drain', drain); response.removeListener('close', close); signal.removeEventListener('abort', abort) }
    const drain = () => { clear(); resolve() }, close = () => { clear(); reject(new Error('The inference client disconnected.')) }, abort = () => { clear(); reject(new Error('Inference was cancelled.')) }
    response.once('drain', drain); response.once('close', close); signal.addEventListener('abort', abort, { once: true }); if (signal.aborted) abort()
  })
}
/** The transport can end only after an actual provider terminal marker, never a short stream. */
export class StreamCompletion {
  private tail = ''
  private terminal = false
  private finished = false
  private failed = false
  push(text: string): void {
    this.tail = (this.tail + text).replace(/\r\n/g, '\n')
    if (this.tail.length > 1024 * 1024) throw new Error('The inference stream event exceeds its byte limit.')
    while (true) {
      const boundary = this.tail.indexOf('\n\n')
      if (boundary < 0) break
      const event = this.tail.slice(0, boundary); this.tail = this.tail.slice(boundary + 2)
      const data = event.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
      if (!data) continue
      if (data.trim() === '[DONE]') { if (!this.finished) throw new Error('The stream ended without a provider completion.'); this.terminal = true; continue }
      let value: any
      try { value = JSON.parse(data) } catch { throw new Error('The inference stream contains malformed event JSON.') }
      if (value.error || value.type === 'response.failed' || value.type === 'error') this.failed = true
      if (Array.isArray(value.choices) && value.choices.some((choice: any) => typeof choice?.finish_reason === 'string' && choice.finish_reason)) this.finished = true
      if (['response.completed', 'response.incomplete'].includes(value.type)) { this.finished = true; this.terminal = true }
    }
  }
  complete(): boolean { return this.finished && this.terminal && !this.failed }
}
export function completeJson(body: any): boolean {
  if (!body || typeof body !== 'object' || body.error) return false
  if (body.object === 'response' || body.type === 'response') return ['completed', 'incomplete'].includes(body.status)
  return Array.isArray(body.choices) && body.choices.length > 0 && body.choices.every((choice: any) => typeof choice?.finish_reason === 'string' && !!choice.finish_reason)
}
