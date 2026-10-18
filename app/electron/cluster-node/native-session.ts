import { spawn, type ChildProcess } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { createServer } from 'node:net'
import { setTimeout as delay } from 'node:timers/promises'
import { terminateOwned } from './command'
import type { EngineSession, EngineStream } from './types'

export async function freeLoopbackPort(): Promise<number> {
  const socket = createServer()
  await new Promise<void>((resolve, reject) => { socket.once('error', reject); socket.listen(0, '127.0.0.1', resolve) })
  const port = (socket.address() as { port: number }).port
  await new Promise<void>((resolve, reject) => socket.close(error => error ? reject(error) : resolve()))
  return port
}
export class NativeSession implements EngineSession {
  private child: ChildProcess
  private ended = false
  private closed: Promise<void>
  private stopping: Promise<void> | null = null
  readonly baseUrl: string
  constructor(binary: string, args: string[], env: NodeJS.ProcessEnv, readonly model: string, private apiKey: string, private keyFile: string, port: number) {
    this.baseUrl = `http://127.0.0.1:${port}`
    this.child = spawn(binary, args, { env, shell: false, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] })
    // Drain logs without returning arbitrary paths, prompts or credentials over management status.
    let bytes = 0
    const drain = (part: Buffer) => { bytes += part.length; if (bytes > 64 * 1024 * 1024) void this.stop().catch(() => {}) }
    this.child.stdout?.on('data', drain); this.child.stderr?.on('data', drain)
    this.closed = new Promise(resolve => { this.child.once('error', () => {}); this.child.once('close', () => { this.ended = true; resolve() }) })
  }
  alive(): boolean { return !this.ended && !!this.child.pid && this.child.exitCode === null && this.child.signalCode === null }
  async ready(signal: AbortSignal, timeoutMs = 15 * 60_000): Promise<void> {
    const deadline = Date.now() + timeoutMs
    while (true) {
      if (signal.aborted) throw new Error('Native model startup cancelled.')
      if (!this.alive()) throw new Error('The owned native engine exited before model readiness.')
      try {
        const response = await fetch(this.baseUrl + '/v1/models', { headers: { authorization: `Bearer ${this.apiKey}` }, redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(2_000)]) })
        const bytes = await readBoundedResponse(response, 64 * 1024)
        const value = JSON.parse(bytes.toString('utf8'))
        if (response.ok && value?.data?.some((item: any) => item?.id === this.model) && this.alive()) return
      } catch { if (signal.aborted) throw new Error('Native model startup cancelled.') }
      if (Date.now() >= deadline) throw new Error('The native engine did not report model readiness before its load deadline.')
      await delay(250, undefined, { signal })
    }
  }
  async request(route: '/v1/chat/completions' | '/v1/responses', body: Record<string, unknown>, signal: AbortSignal): Promise<EngineStream> {
    if (!this.alive()) throw new Error('The owned native engine is no longer running.')
    let response: Response
    try { response = await fetch(this.baseUrl + route, { method: 'POST', headers: { authorization: `Bearer ${this.apiKey}`, 'content-type': 'application/json' }, body: JSON.stringify({ ...body, model: this.model }), redirect: 'error', signal }) }
    catch (error) { await this.settleRequest(); throw error }
    if (!response.body) { await this.settleRequest(); throw new Error('The native engine returned an empty response stream.') }
    const upstream = response.body, owner = this
    async function* streamBody(): AsyncIterable<Uint8Array> {
      try { for await (const part of upstream as unknown as AsyncIterable<Uint8Array>) yield part }
      finally { await owner.settleRequest() }
    }
    return { status: response.status, contentType: response.headers.get('content-type') ?? 'application/json', body: streamBody() }
  }
  private async settleRequest(): Promise<void> {
    const deadline = Date.now() + 10_000
    while (this.alive()) {
      try {
        const response = await fetch(this.baseUrl + '/stats', { headers: { authorization: `Bearer ${this.apiKey}` }, redirect: 'error', signal: AbortSignal.timeout(2_000) })
        const value = JSON.parse((await readBoundedResponse(response, 64 * 1024)).toString('utf8'))
        if (response.ok && value.running === 0 && value.waiting === 0) return
      } catch { /* No idle acknowledgement means no new physical inference admission. */ }
      if (Date.now() >= deadline) { await this.stop(); throw new Error('The native request did not confirm idle after transport completion; its owned process was stopped.') }
      await delay(100)
    }
  }

  stop(): Promise<void> {
    if (this.stopping) return this.stopping
    this.stopping = (async () => {
      if (!this.ended) {
        terminateOwned(this.child, 'SIGTERM')
        const hard = setTimeout(() => terminateOwned(this.child, 'SIGKILL'), 5_000)
        let overdue: ReturnType<typeof setTimeout> | undefined
        try { await Promise.race([this.closed, new Promise<never>((_resolve, reject) => { overdue = setTimeout(() => reject(new Error('The owned native process has not physically exited; review it before another load.')), 10_000) })]) }
        finally { clearTimeout(hard); if (overdue) clearTimeout(overdue) }
      }
      await fs.unlink(this.keyFile).catch(() => {})
    })()
    return this.stopping
  }
}
export async function readBoundedResponse(response: Response, maximum: number): Promise<Buffer> {
  const length = Number(response.headers.get('content-length'))
  if (Number.isFinite(length) && length > maximum) { await response.body?.cancel(); throw new Error('The native response exceeds its byte limit.') }
  if (!response.body) return Buffer.alloc(0)
  const chunks: Buffer[] = []; let bytes = 0
  const reader = response.body.getReader()
  try { while (true) { const next = await reader.read(); if (next.done) break; bytes += next.value.byteLength; if (bytes > maximum) throw new Error('The native response exceeds its byte limit.'); chunks.push(Buffer.from(next.value)) } }
  catch (error) { await reader.cancel().catch(() => {}); throw error }
  finally { reader.releaseLock() }
  return Buffer.concat(chunks, bytes)
}
