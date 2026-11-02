import type { IncomingMessage, ServerResponse } from 'node:http'
import { once } from 'node:events'
import { readJson, writeJson } from '../remote-control-http'
import type { AskGeneration, AskStep } from './generate'
import type { AskSavedState } from './session'
import { safeAskError } from './session'
import type { QuickAskRoute } from '../../../shared/quick-ask'
import type { Thread } from '../types'

export interface QuickAskRemoteBackend {
  routes(): Promise<QuickAskRoute[]>
  generate(input: AskGeneration, signal: AbortSignal): AsyncIterable<AskStep>
  load(routeId: string): Promise<void>
  handoff(state: AskSavedState): Promise<Thread>
}
export async function handleQuickAskRemote(req: IncomingMessage, res: ServerResponse, pathname: string, owner: boolean, authorized: () => boolean, backend?: QuickAskRemoteBackend): Promise<boolean> {
  if (!pathname.startsWith('/v1/ask/')) return false
  if (!owner) { writeJson(res, 403, { error: 'owner_token_required' }); return true }
  if (!backend) { writeJson(res, 501, { error: 'Update the TensorFold host to use Ask TensorFold.' }); return true }
  if (!authorized()) { writeJson(res, 401, { error: 'unauthorized' }); return true }
  if (req.method === 'GET' && pathname === '/v1/ask/routes') { writeJson(res, 200, await backend.routes()); return true }
  if (req.method !== 'POST') { writeJson(res, 404, { error: 'not_found' }); return true }
  const value = await readJson(req)
  if (!authorized()) { writeJson(res, 401, { error: 'unauthorized' }); return true }
  if (pathname === '/v1/ask/load') { await backend.load(String(value.routeId ?? '')); writeJson(res, 200, { ok: true }); return true }
  if (pathname === '/v1/ask/handoff') { writeJson(res, 200, await backend.handoff(value as unknown as AskSavedState)); return true }
  if (pathname !== '/v1/ask/stream') { writeJson(res, 404, { error: 'not_found' }); return true }
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 180_000)
  const revoke = setInterval(() => { if (!authorized()) controller.abort() }, 1000)
  const disconnected = () => controller.abort()
  res.once('close', disconnected)
  res.writeHead(200, { 'content-type': 'application/x-ndjson', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' })
  const write = async (step: AskStep | { type: 'error'; error: string }) => {
    controller.signal.throwIfAborted()
    if (!authorized()) throw new Error('The host authorization changed.')
    if (!res.write(`${JSON.stringify(step)}\n`)) await once(res, 'drain', { signal: controller.signal })
  }
  try {
    for await (const step of backend.generate(value as unknown as AskGeneration, controller.signal)) await write(step)
  } catch (error) {
    if (!res.destroyed && !controller.signal.aborted && authorized()) await write({ type: 'error', error: safeAskError(error) }).catch(() => {})
  } finally {
    clearTimeout(timeout); clearInterval(revoke); controller.abort(); res.removeListener('close', disconnected); res.end()
  }
  return true
}
