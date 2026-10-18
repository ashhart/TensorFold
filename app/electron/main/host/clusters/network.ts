import { createSshHttpRequest } from './ssh-http'
import { createHash } from 'node:crypto'
import type { SshHost } from '../../../agent/types'
import type { SshSession } from '../../../agent/ssh-connection'
import type { StoredClusterHost } from './store'
import { ClusterNodeClient } from './client'
import { createClusterSshForward } from './ssh-forward'

export function clusterSshStamp(host: SshHost): string {
  return createHash('sha256').update(JSON.stringify([host.id, host.hostName, host.user, host.port, host.identityFiles, host.proxyJump, host.forwardAgent, host.configAlias, host.updatedAt])).digest('hex')
}
export interface ClusterNetworkOptions {
  assertCurrent(): void
  hostCurrent(host: StoredClusterHost): Promise<void>
  resolveSshHost(id: string): Promise<SshHost | null>
  sessionForHost(id: string): Promise<SshSession>
}
/** Each credential is bound to both the saved target and the actual verified SSH session. */
export function createClusterNetwork(options: ClusterNetworkOptions) {
  const forward = createClusterSshForward({ sessionForHost: options.sessionForHost })
  const leases = new Map<string, ReturnType<typeof forward.open>>()
  const epochs = new Map<string, number>()
  const sessionIds = new WeakMap<SshSession, number>(), unsubscribes = new Map<string, () => void>()
  let nextSessionId = 0
  const closedSessions = new WeakSet<SshSession>()
  let closed = false
  const controller = new AbortController()
  const current = () => { options.assertCurrent(); if (closed) throw new Error('The cluster connection scope closed.') }
  return {
    async stamp(id: string) {
      current(); const host = await options.resolveSshHost(id); current()
      if (!host) throw new Error('Choose an existing saved SSH host.')
      return clusterSshStamp(host)
    },
    async client(host: StoredClusterHost): Promise<ClusterNodeClient> {
      current(); if (!host.token) throw new Error('Add this machine’s management credential before inspection.')
      await options.hostCurrent(host); current()
      let baseUrl: string
      let session: SshSession | null = null
      if (host.connection.kind === 'endpoint') baseUrl = host.connection.url
      else {
        const { sshHostId, port } = host.connection
        const saved = await options.resolveSshHost(sshHostId); current()
        if (!saved || !host.sshStamp || clusterSshStamp(saved) !== host.sshStamp) throw new Error('The saved SSH target changed; enroll its management credential again.')
        session = await options.sessionForHost(sshHostId); current()
        if (clusterSshStamp(session.host) !== host.sshStamp) throw new Error('The connected SSH session belongs to a different saved target; reconnect and enroll again.')
        if (!sessionIds.has(session)) sessionIds.set(session, ++nextSessionId)
        const key = `${host.id}:${host.revision}:${host.sshStamp}:${sessionIds.get(session)}`
        let opening = leases.get(key)
        if (!opening) {
          const exactSession = session, epoch = epochs.get(host.id) ?? 0
          let sessionClosed = false
          opening = forward.open(sshHostId, port, { session: exactSession, assertCurrent: () => { current(); if (sessionClosed || (epochs.get(host.id) ?? 0) !== epoch || clusterSshStamp(exactSession.host) !== host.sshStamp) throw new Error('The SSH transport identity changed.') } })
          leases.set(key, opening)
          const captured = opening
          const unsubscribe = exactSession.onClosed(() => {
            sessionClosed = true; closedSessions.add(exactSession)
            if (leases.get(key) === captured) leases.delete(key)
            unsubscribes.get(key)?.(); unsubscribes.delete(key)
            void captured.then(lease => lease.close()).catch(() => {})
          })
          unsubscribes.set(key, unsubscribe)
          if (sessionClosed) { unsubscribe(); unsubscribes.delete(key) }
          void opening.catch(() => { if (leases.get(key) === opening) { leases.delete(key); unsubscribes.get(key)?.(); unsubscribes.delete(key) } })
        }
        const lease = await opening
        current()
        baseUrl = lease.url
      }
      const exactSession = session
      const assertCurrent = async () => {
        current(); await options.hostCurrent(host); current()
        if (host.connection.kind === 'ssh') {
          const saved = await options.resolveSshHost(host.connection.sshHostId); current()
          if (!saved || clusterSshStamp(saved) !== host.sshStamp || !exactSession || closedSessions.has(exactSession) || clusterSshStamp(exactSession.host) !== host.sshStamp) throw new Error('The SSH target changed; no credential was sent.')
        }
      }
      await assertCurrent()
      let request: ((url: string, init: RequestInit) => Promise<Response>) | undefined
      if (exactSession && host.connection.kind === 'ssh') {
        // A transport is request-scoped so its guard cannot accidentally capture another host.
        request = async (url, init) => {
          const transport = createSshHttpRequest(exactSession, host.connection.kind === 'ssh' ? host.connection.port : 0, assertCurrent)
          try { return await transport.request(url, { ...init, signal: init.signal ? AbortSignal.any([init.signal, controller.signal]) : controller.signal }) } finally { transport.close() }
        }
      }
      return new ClusterNodeClient({ baseUrl, token: host.token, nodeId: host.nodeId, assertCurrent, request })
    },
    async verify(input: { sshHostId: string; port: number; token: string; session: SshSession; signal?: AbortSignal; assertCurrent(): void }) {
      current(); input.assertCurrent()
      const transport = createSshHttpRequest(input.session, input.port, () => { current(); input.assertCurrent() })
      try {
        const client = new ClusterNodeClient({ baseUrl: 'http://127.0.0.1', token: input.token, nodeId: null, assertCurrent: () => { current(); input.assertCurrent() }, request: transport.request })
        return { protocol: 1 as const, nodeId: await client.identity(input.signal), authenticated: true as const }
      } finally { transport.close() }
    },
    async consumer(host: StoredClusterHost, url: string, init: RequestInit) {
      const client = await this.client(host)
      if (new URL(url).origin !== new URL(client.baseUrl).origin || !client.options.request) throw new Error('Reconnect this verified SSH endpoint before sending work.')
      return client.options.request(url, init)
    },
    async invalidate(hostId: string) {
      epochs.set(hostId, (epochs.get(hostId) ?? 0) + 1)
      const targets = [...leases].filter(([key]) => key.startsWith(hostId + ':'))
      for (const [key] of targets) { leases.delete(key); unsubscribes.get(key)?.(); unsubscribes.delete(key) }
      await Promise.allSettled(targets.map(async ([, value]) => { const lease = await value; await lease.close() }))
    },
    async close() { closed = true; controller.abort(); for (const unsubscribe of unsubscribes.values()) unsubscribe(); unsubscribes.clear(); const pending = [...leases.values()]; leases.clear(); await Promise.allSettled(pending.map(async item => { const lease = await item; await lease.close() })) },
  }
}
