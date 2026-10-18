import net from 'node:net'
import type { Duplex } from 'node:stream'

export interface ClusterForwardSession {
  forwardLoopback(port: number, signal?: AbortSignal, timeoutMs?: number): Promise<Duplex>
  onClosed(listener: () => void): () => void
}
export interface ClusterForwardLease {
  url: string; location: 'remote'; sshHostId: string; remotePort: number
  close(): Promise<void>
}
export interface ClusterForwardDeps {
  sessionForHost(id: string): Promise<ClusterForwardSession>
  createServer?: typeof net.createServer
  maxConnections?: number
}

/** A verified SSH transport remains remote even though its client listener is loopback. */
export function createClusterSshForward(deps: ClusterForwardDeps) {
  return {
    async open(sshHostId: string, remotePort: number, options: { signal?: AbortSignal; assertCurrent?: () => void; session?: ClusterForwardSession } = {}): Promise<ClusterForwardLease> {
      if (typeof sshHostId !== 'string' || !sshHostId || sshHostId.length > 320 || !Number.isInteger(remotePort) || remotePort < 1 || remotePort > 65_535) throw new Error('Choose a saved SSH host and valid loopback port.')
      const current = () => { if (options.signal?.aborted) throw new Error('SSH forwarding was cancelled.'); options.assertCurrent?.() }
      current(); const session = options.session ?? await deps.sessionForHost(sshHostId); current()
      const connections = new Set<net.Socket>(), channels = new Set<Duplex>(), opening = new Set<Promise<void>>()
      const controller = new AbortController()
      let closed = false, closedPromise: Promise<void> | null = null, listeningSettled: Promise<void> | null = null, removeSession = () => {}
      const server = (deps.createServer ?? net.createServer)(socket => {
        if (closed || connections.size >= Math.max(1, Math.min(64, deps.maxConnections ?? 16))) { socket.destroy(); return }
        try { current() } catch { socket.destroy(); return }
        connections.add(socket)
        socket.setTimeout(60_000, () => socket.destroy())
        let channel: Duplex | null = null
        const dispose = () => { connections.delete(socket); if (channel) { channels.delete(channel); channel.destroy(); channel = null } }
        socket.once('close', dispose); socket.once('error', dispose)
        const attempt = session.forwardLoopback(remotePort, controller.signal).then(opened => {
          channel = opened
          try { current() } catch { opened.destroy(); socket.destroy(); return }
          if (closed || socket.destroyed) { opened.destroy(); socket.destroy(); return }
          channels.add(opened)
          opened.once('error', () => socket.destroy()); opened.once('close', () => { channels.delete(opened); socket.destroy() })
          socket.pipe(opened).pipe(socket)
        }).catch(() => { socket.destroy() }).finally(() => { opening.delete(attempt) })
        opening.add(attempt)
      })
      const close = (): Promise<void> => {
        if (closedPromise) return closedPromise
        closed = true; controller.abort(); removeSession(); options.signal?.removeEventListener('abort', abort)
        for (const socket of connections) socket.destroy()
        for (const channel of channels) channel.destroy()
        closedPromise = (async () => {
          await listeningSettled?.catch(() => {})
          await Promise.allSettled([...opening])
          await new Promise<void>(resolve => { if (!server.listening) { resolve(); return }; server.close(() => resolve()) })
        })()
        return closedPromise
      }
      const abort = () => { void close() }
      removeSession = session.onClosed(abort)
      options.signal?.addEventListener('abort', abort, { once: true })
      if (options.signal?.aborted || closed) { await close(); throw new Error('SSH forwarding was cancelled or its session closed.') }
      try {
        listeningSettled = new Promise<void>((resolve, reject) => {
          const failed = (error: Error) => { server.removeListener('listening', ready); reject(error) }
          const ready = () => { server.removeListener('error', failed); resolve() }
          server.once('error', failed); server.once('listening', ready)
          server.listen({ host: '127.0.0.1', port: 0, exclusive: true })
        })
        await listeningSettled
        current()
        if (closed) throw new Error('SSH connection closed before its forward was ready.')
        const address = server.address()
        if (!address || typeof address === 'string' || address.address !== '127.0.0.1') throw new Error('SSH forwarding did not bind exclusively to loopback.')
        server.on('error', abort)
        return { url: `http://127.0.0.1:${address.port}`, location: 'remote', sshHostId, remotePort, close }
      } catch (error) { await close(); throw error }
    },
  }
}
