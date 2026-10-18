import { OsNodePublisher, type NodeAdvertisement } from './advertise'
import { acquireNodeLease } from './lease'
import { createHmac } from 'node:crypto'
import { createServer as createHttpServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { createServer as createHttpsServer } from 'node:https'
import { promises as fs, constants } from 'node:fs'
import type { Socket } from 'node:net'
import { clusterId, clusterObject, validateClusterCommandId, validateClusterGeneration, validateClusterModel, validateClusterNodeLaunch } from '../../shared/cluster-validation'
import { authorized, createOrReadTokens, isLoopback, NodeHttpError, privateRead, publicMessage, validateNodeConfig } from './config'
import { DiskNodeJournal } from './journal'
import { ClusterNodeRuntime } from './runtime'
import { NodeEngineAdapter } from './native-adapter'
import { completeJson, json, readJson, StreamCompletion, writeChunk } from './http'
import type { ClusterNodeServer, NodeAdmissionInput, NodeServerOptions } from './types'

const PREFIX = '/api/cluster/v1'
function command(value: unknown, extras: string[]) {
  const input = clusterObject(value, ['commandId', 'expectedNodeId', ...extras])
  return { input, commandId: validateClusterCommandId(input.commandId), expectedNodeId: clusterId(input.expectedNodeId) }
}
function admissionInput(value: unknown): NodeAdmissionInput {
  const raw = clusterObject(value, ['expectedNodeId', 'role', 'model', 'profileId', 'generation', 'transport', 'rank', 'worldSize', 'links'])
  const parsed = validateClusterNodeLaunch({ ...raw, deploymentId: 'admission-only', commandId: 'admission-only' })
  const { deploymentId: _deploymentId, commandId: _commandId, ...input } = parsed
  return input
}
export async function createClusterNodeServer(options: NodeServerOptions): Promise<ClusterNodeServer> {
  const config = validateNodeConfig(options.config)
  const ownerLease = await acquireNodeLease(config.dataDir)
  try {
  const maxBody = Math.max(1, Math.min(options.maxBodyBytes ?? 1024 * 1024, 8 * 1024 * 1024)), maxResponse = Math.max(1, Math.min(options.maxResponseBytes ?? 64 * 1024 * 1024, 128 * 1024 * 1024))
  const timeout = Math.max(10, Math.min(options.requestTimeoutMs ?? 15 * 60_000, 30 * 60_000))
  const tokens = await createOrReadTokens(config.dataDir, options.tokenFiles)
  const runtime = new ClusterNodeRuntime(options.journal ?? new DiskNodeJournal(config.dataDir), options.engine ?? new NodeEngineAdapter({ config }), options.now)
  await runtime.initialize()
  const sockets = new Set<Socket>(), authFailures = new Map<string, { at: number; failures: number }>()
  let advertisement: NodeAdvertisement | null = null, publishing: Promise<NodeAdvertisement> | null = null
  const lifecycle = new AbortController()
  let endpoint: { host: string; port: number; url: string } | null = null, closing: Promise<void> | null = null, requests = 0
  const inferenceKey = (active: { id: string; commandId: string } | null) => active ? createHmac('sha256', Buffer.from(tokens.inference, 'hex')).update(active.id).update('\0').update(active.commandId).digest('hex') : tokens.inference
  const authenticate = async (request: IncomingMessage, management: boolean) => {
    if (request.headers.origin) throw new NodeHttpError(403, 'Browser-origin requests cannot invoke this management or inference service.')
    const key = request.socket.remoteAddress ?? 'unknown', now = options.now?.() ?? Date.now(), prior = authFailures.get(key)
    if (prior && now - prior.at < 60_000 && prior.failures >= 12) throw new NodeHttpError(429, 'Too many rejected authentication attempts; wait before retrying.')
    const active = management ? null : (await runtime.state()).active
    if (!authorized(request.headers.authorization, management ? tokens.management : inferenceKey(active))) {
      if (authFailures.size >= 128 && !authFailures.has(key)) authFailures.delete(authFailures.keys().next().value!)
      authFailures.set(key, { at: prior && now - prior.at < 60_000 ? prior.at : now, failures: prior && now - prior.at < 60_000 ? prior.failures + 1 : 1 })
      throw new NodeHttpError(401, 'A valid service credential is required.')
    }
    return active ? { deploymentId: active.id, commandId: active.commandId } : null
  }
  const proxy = async (request: IncomingMessage, response: ServerResponse, route: '/v1/chat/completions' | '/v1/responses', binding: { deploymentId: string; commandId: string } | null) => {
    const body = await readJson(request, maxBody)
    if (typeof body.model !== 'string' || body.model.length > 160 || body.stream !== undefined && typeof body.stream !== 'boolean') throw new NodeHttpError(400, 'Choose the current deployment model and valid streaming settings.')
    if (body.n !== undefined && body.n !== 1) throw new NodeHttpError(400, 'This node admits one completion per request.')
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeout)
    const abort = () => { if (!response.writableFinished) controller.abort() }
    request.on('aborted', abort); response.on('close', abort)
    let lease: Awaited<ReturnType<ClusterNodeRuntime['requestLease']>> | null = null
    try {
      if (!binding) throw new NodeHttpError(409, 'No authenticated deployment is available.')
      lease = await runtime.requestLease(body, controller.signal, false, binding)
      const stream = await lease.request(route), sse = /^text\/event-stream(?:;|$)/i.test(stream.contentType)
      if (!sse) {
        const parts: Buffer[] = []; let bytes = 0
        for await (const part of stream.body) { bytes += part.byteLength; if (bytes > maxResponse) throw new Error('The inference response exceeds its byte limit.'); parts.push(Buffer.from(part)) }
        const data = Buffer.concat(parts, bytes), value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(data))
        const complete = stream.status < 400 && completeJson(value)
        if (stream.status < 400 && !complete) throw new Error('The inference response was partial or malformed.')
        await lease.finish(complete ? 'completed' : 'failed')
        response.writeHead(stream.status, { 'content-type': 'application/json', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' }); response.end(data)
      } else {
        if (stream.status >= 400) throw new Error('The native engine refused its streaming response.')
        const completion = new StreamCompletion(), decoder = new TextDecoder('utf-8', { fatal: true }); let bytes = 0
        response.writeHead(stream.status, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' })
        for await (const part of stream.body) { bytes += part.byteLength; if (bytes > maxResponse) throw new Error('The inference stream exceeds its byte limit.'); completion.push(decoder.decode(part, { stream: true })); await writeChunk(response, part, controller.signal) }
        completion.push(decoder.decode())
        if (!completion.complete()) throw new Error('The inference stream ended without a provider completion; no finished result was accepted.')
        await lease.finish('completed'); response.end()
      }
    } catch (error) {
      controller.abort()
      await lease?.finish(controller.signal.aborted && response.destroyed ? 'cancelled' : 'failed').catch(() => {})
      if (response.headersSent) response.destroy(); else throw error instanceof NodeHttpError ? error : new NodeHttpError(502, 'The native response was incomplete, invalid or outside its request bounds.')
    } finally { clearTimeout(timer); request.removeListener('aborted', abort); response.removeListener('close', abort) }
  }
  const route = async (request: IncomingMessage, response: ServerResponse) => {
    if (++requests > 32) { requests--; json(response, 429, { error: 'The node request limit is busy.' }); return }
    try {
      if (!request.url?.startsWith('/') || request.url.startsWith('//') || request.url.length > 2048) throw new NodeHttpError(400, 'Choose a bounded node route.')
      const url = new URL(request.url, 'http://node.invalid'), pathname = url.pathname, admin = pathname.startsWith(PREFIX + '/')
      const binding = await authenticate(request, admin)
      if (admin) {
        if (request.method === 'GET' && pathname === PREFIX + '/identity') {
          const nonce = url.searchParams.get('nonce')
          if (!nonce || !/^[A-Za-z0-9_-]{8,160}$/.test(nonce) || [...url.searchParams.keys()].some(key => key !== 'nonce')) throw new NodeHttpError(400, 'Choose a bounded identity nonce.')
          return json(response, 200, { protocol: 1, nodeId: (await runtime.state()).nodeId, nonce, authenticated: true })
        }
        if (url.search) throw new NodeHttpError(400, 'Unexpected query parameters.')
        if (request.method === 'GET' && pathname === PREFIX + '/status') return json(response, 200, await runtime.snapshot(isLoopback(config.host)))
        if (request.method === 'GET' && pathname === PREFIX + '/endpoint-credentials') {
          const snapshot = await runtime.snapshot(isLoopback(config.host))
          if (!snapshot.endpoint?.ready || !snapshot.active?.canaryPassed) throw new NodeHttpError(409, 'The owned deployment must be ready and pass its canary before credentials are exposed.')
          const active = (await runtime.state()).active
          if (!active || active.id !== snapshot.active.id || active.commandId !== snapshot.active.commandId || active.state !== 'ready' || !active.canaryPassed) throw new NodeHttpError(409, 'The owned deployment changed before credential retrieval.')
          return json(response, 200, { apiKey: inferenceKey(active), model: active.model.repo })
        }
        if (request.method === 'GET' && pathname === PREFIX + '/jobs') return json(response, 200, { jobs: (await runtime.state()).jobs })
        const jobRoute = new RegExp(`^${PREFIX}/jobs/([A-Za-z0-9_-]{1,160})(/cancel)?$`).exec(pathname)
        if (jobRoute) {
          if (request.method === 'GET' && !jobRoute[2]) { const job = (await runtime.state()).jobs.find(job => job.id === jobRoute[1]); if (!job) throw new NodeHttpError(404, 'This node job was not found.'); return json(response, 200, job) }
          if (request.method === 'POST' && jobRoute[2]) { const value = command(await readJson(request, maxBody), []); return json(response, 202, await runtime.cancel(jobRoute[1], value)) }
        }
        if (request.method !== 'POST') throw new NodeHttpError(404, 'This node route was not found.')
        const body = await readJson(request, maxBody)
        if (pathname === PREFIX + '/admission') return json(response, 200, await runtime.admission(admissionInput(body)))
        if (pathname === PREFIX + '/models/ensure') {
          const value = command(body, ['model'])
          return json(response, 202, await runtime.accept('ensure-model', { commandId: value.commandId, expectedNodeId: value.expectedNodeId, model: validateClusterModel(value.input.model) }))
        }
        if (pathname === PREFIX + '/deployments/start') return json(response, 202, await runtime.accept('start', validateClusterNodeLaunch(body)))
        if (pathname === PREFIX + '/deployments/stop' || pathname === PREFIX + '/canary') {
          const value = command(body, ['deploymentId']), input = { commandId: value.commandId, expectedNodeId: value.expectedNodeId, deploymentId: clusterId(value.input.deploymentId) }
          return json(response, 202, await runtime.accept(pathname.endsWith('/canary') ? 'canary' : 'stop', input))
        }
      } else {
        if (url.search) throw new NodeHttpError(400, 'Unexpected inference query parameters.')
        if (request.method === 'GET' && pathname === '/v1/models') {
          const snapshot = await runtime.snapshot(isLoopback(config.host))
          if (!binding || snapshot.active?.id !== binding.deploymentId || snapshot.active.commandId !== binding.commandId) throw new NodeHttpError(409, 'The authenticated deployment changed.')
          if (!snapshot.endpoint?.ready) throw new NodeHttpError(409, 'The deployment is not ready and canary-qualified.')
          return json(response, 200, { object: 'list', data: [{ id: snapshot.endpoint.model, object: 'model', owned_by: 'tensorfold' }] })
        }
        if (request.method === 'POST' && (pathname === '/v1/chat/completions' || pathname === '/v1/responses')) return await proxy(request, response, pathname, binding)
      }
      throw new NodeHttpError(404, 'This node route was not found.')
    } catch (error) {
      if (response.headersSent) response.destroy()
      else json(response, error instanceof NodeHttpError ? error.status : 400, { error: publicMessage(error) })
    } finally { requests-- }
  }
  let server
  if (config.tls) {
    const key = await privateRead(config.tls.keyFile, 64 * 1024), handle = await fs.open(config.tls.certFile, constants.O_RDONLY | constants.O_NOFOLLOW)
    let cert: Buffer
    try { const stat = await handle.stat(); if (!stat.isFile() || stat.size > 64 * 1024) throw new Error('The TLS certificate exceeds its bounded file limit.'); cert = await handle.readFile() } finally { await handle.close() }
    server = createHttpsServer({ cert, key, minVersion: 'TLSv1.2' }, (request, response) => { void route(request, response) })
  } else server = createHttpServer((request, response) => { void route(request, response) })
  server.maxHeadersCount = 32; server.headersTimeout = 10_000; server.requestTimeout = 30_000; server.keepAliveTimeout = 5_000
  server.on('connection', socket => { if (sockets.size >= 128) { socket.destroy(); return }; sockets.add(socket); socket.once('close', () => sockets.delete(socket)) })
  const api: ClusterNodeServer = {
    server,
    address: () => endpoint,
    snapshot: () => runtime.snapshot(isLoopback(config.host)),
    listen: async () => {
      if (closing) throw new NodeHttpError(503, 'This node server has already stopped.')
      if (endpoint) return endpoint
      await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(config.port, config.host, () => { server.removeListener('error', reject); resolve() }) })
      const port = (server.address() as { port: number }).port, host = config.host.includes(':') ? `[${config.host}]` : config.host
      endpoint = { host: config.host, port, url: `${config.tls ? 'https' : 'http'}://${host}:${port}` }
      if (config.advertise) {
        publishing = (options.publisher ?? new OsNodePublisher({ diagnostic: options.diagnostic })).start((await runtime.state()).nodeId, port, lifecycle.signal, config.tls ? 'https' : 'ssh')
        try { advertisement = await publishing } catch (error) { await api.close(); throw error }
      }
      return endpoint
    },
    close: () => closing ??= (async () => {
      lifecycle.abort()
      const stopped = new Promise<void>((resolve, reject) => { if (!server.listening) { resolve(); return }; server.close(error => error ? reject(error) : resolve()) })
      let failure: unknown
      try { await runtime.close() } catch (error) { failure = error }
      try { if (publishing) advertisement = await publishing.catch(() => null); await advertisement?.stop() } catch (error) { failure ??= error }
      finally { for (const socket of sockets) socket.destroy(); await stopped; endpoint = null; await ownerLease.release() }
      if (failure) throw failure
    })(),
  }
  return api
  } catch (error) { await ownerLease.release(); throw error }
}
