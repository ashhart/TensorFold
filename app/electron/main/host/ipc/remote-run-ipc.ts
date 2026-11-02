import { clientVisibleRemoteSessions, stampRemoteTerminalSession } from '../../../agent/terminal-forward'
import type { CliSession } from '../../../agent/types'
import type { CliDefinition } from '../../../agent/types/cli'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { assertGmailInboxRoute } from '../../../../shared/gmail-inbox'
import type { RemoteIpcContext } from './remote-ipc-context'

export function registerRemoteRunIpc(context: RemoteIpcContext): void {
  const { remoteHostForId, ensureRemoteEventStream, fetchRemoteOrchestratorJson, remoteOrchestratorConnectionError } = context
  handle('remote:listClis', async (_e, input?: { hostId?: string | null }): Promise<CliDefinition[]> => {
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      const { data } = await fetchRemoteOrchestratorJson<CliDefinition[]>(host, '/v1/clis', { signal: AbortSignal.timeout(8000) })
      return data ?? []
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:terminalStart', async (_e, input: { hostId?: string | null; projectId: string; cliId: string; command?: string; label?: string }): Promise<CliSession> => {
    requireSecurityRole('operator', 'start an Orchestrator terminal')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<CliSession>(host, '/v1/terminals', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ projectId: input.projectId, cliId: input.cliId, label: input.label }),
        signal: AbortSignal.timeout(12000),
      })

      return stampRemoteTerminalSession(data, host.id)
    } catch (e) { throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl)) }
  })
  handle('remote:terminalWrite', async (_e, input: { hostId?: string | null; sessionId: string; data: string }): Promise<void> => {
    requireSecurityRole('operator', 'write to an Orchestrator terminal')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    await fetchRemoteOrchestratorJson(host, `/v1/terminals/${encodeURIComponent(input.sessionId)}/write`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ data: input.data }), signal: AbortSignal.timeout(8000),
    })
  })
  handle('remote:terminalResize', async (_e, input: { hostId?: string | null; sessionId: string; cols: number; rows: number }): Promise<void> => {
    requireSecurityRole('operator', 'resize an Orchestrator terminal')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    await fetchRemoteOrchestratorJson(host, `/v1/terminals/${encodeURIComponent(input.sessionId)}/resize`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ cols: input.cols, rows: input.rows }), signal: AbortSignal.timeout(8000),
    })
  })
  handle('remote:terminalRename', async (_e, input: { hostId?: string | null; sessionId: string; title: string }): Promise<void> => {
    requireSecurityRole('operator', 'rename an Orchestrator terminal')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    await fetchRemoteOrchestratorJson(host, `/v1/terminals/${encodeURIComponent(input.sessionId)}/rename`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ title: input.title }), signal: AbortSignal.timeout(8000),
    })
  })
  handle('remote:terminalKill', async (_e, input: { hostId?: string | null; sessionId: string }): Promise<void> => {
    requireSecurityRole('operator', 'kill an Orchestrator terminal process')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    await fetchRemoteOrchestratorJson(host, `/v1/terminals/${encodeURIComponent(input.sessionId)}/kill`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(8000),
    })
  })
  handle('remote:terminalList', async (_e, input: { hostId?: string | null; projectId: string }): Promise<CliSession[]> => {
    requireSecurityRole('operator', 'list Orchestrator terminals')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    const { data } = await fetchRemoteOrchestratorJson<CliSession[]>(host, `/v1/terminals?projectId=${encodeURIComponent(input.projectId)}`, { signal: AbortSignal.timeout(8000) })
    return clientVisibleRemoteSessions(data ?? [], host.id) as CliSession[]
  })

  handle('remote:approve', async (_e, input: { hostId?: string | null; callId: string; decision: 'allow' | 'allow_always' | 'deny' }): Promise<void> => {
    requireSecurityRole('operator', 'resolve Orchestrator approvals')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      await fetchRemoteOrchestratorJson(host, `/v1/approvals/${encodeURIComponent(String(input.callId))}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ decision: input.decision }),
        signal: AbortSignal.timeout(12000),
      })
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:startThread', async (_e, input: { hostId?: string | null; payload: { threadId: string; userText: string; attachments?: unknown[]; sourceThreadKey?: string } }): Promise<{ runId: string }> => {
    requireSecurityRole('operator', 'start Orchestrator thread')
    assertGmailInboxRoute(input?.payload?.sourceThreadKey, true)
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const threadId = encodeURIComponent(String(input.payload.threadId))
      const { data } = await fetchRemoteOrchestratorJson<{ runId: string }>(host, `/v1/threads/${threadId}/start`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.payload),
        signal: AbortSignal.timeout(12000),
      })
      return data
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:guideThread', async (_e, input: { hostId?: string | null; payload: { threadId: string; userText: string; attachments?: unknown[] } }): Promise<{ queued: boolean; queueDepth: number }> => {
    requireSecurityRole('operator', 'guide Orchestrator thread')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const threadId = encodeURIComponent(String(input.payload.threadId))
      const { data } = await fetchRemoteOrchestratorJson<{ queued: boolean; queueDepth: number }>(host, `/v1/threads/${threadId}/guide`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.payload),
        signal: AbortSignal.timeout(12000),
      })
      return data
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:cancelThread', async (_e, input: { hostId?: string | null; threadId: string }): Promise<void> => {
    requireSecurityRole('operator', 'cancel Orchestrator thread')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      await fetchRemoteOrchestratorJson(host, `/v1/threads/${encodeURIComponent(String(input.threadId))}/cancel`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(12000),
      })
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

}
