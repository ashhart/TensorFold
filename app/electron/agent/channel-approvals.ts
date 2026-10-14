// In-channel approval plumbing for the connector-action gate. Two concerns, kept pure so
// they unit-test without a network:
//   1) A pending-approval REGISTRY: requestApproval posts an inline prompt and parks a
//      Promise keyed by a short token; an inbound decision (button callback or a "/approve"
//      reply) resolves it — "block until they answer", mirroring the in-app gate.
//   2) Inline-control MARKUP builders per platform (Slack Block Kit, Discord components,
//      Telegram inline keyboard). The approve/deny buttons carry the token so the inbound
//      callback can resolve the exact pending request.
//
// The token is opaque and short; it never carries user content, so it is safe to embed in a
// button payload that round-trips through the platform.

import { randomUUID } from 'node:crypto'
import type { ChannelApprovalRequest } from './channel-adapter'
import type { ConnectorApprovalDecision } from './types/connectors'

export interface PendingApproval {
  token: string
  platform: string
  threadKey: string
  createdAt: number
  resolve: (decision: ConnectorApprovalDecision) => void
}

interface PendingEntry extends PendingApproval {
  timer: ReturnType<typeof setTimeout> | null
  settled: boolean
}

// Default: a parked approval that no one answers resolves to 'deny' after this long, so an
// agent run never blocks forever waiting on a channel that went quiet (fail-safe = deny).
export const DEFAULT_APPROVAL_TIMEOUT_MS = 10 * 60_000

export interface ChannelApprovalRegistry {
  /** Park a pending approval; returns the token to embed in inline controls and the promise. */
  create(platform: string, threadKey: string): { token: string; promise: Promise<ConnectorApprovalDecision> }
  /** Resolve a parked approval by token (from a button callback). Returns true if one matched. */
  resolve(token: string, decision: ConnectorApprovalDecision): boolean
  /**
   * Resolve the newest unsettled approval for a conversation by threadKey (from a plain
   * "/approve" reply that carries no token). Returns true if one matched.
   */
  resolveByThreadKey(threadKey: string, decision: ConnectorApprovalDecision): boolean
  /** Number of still-parked approvals (for diagnostics/tests). */
  size(): number
}

export function createChannelApprovalRegistry(timeoutMs = DEFAULT_APPROVAL_TIMEOUT_MS): ChannelApprovalRegistry {
  const pending = new Map<string, PendingEntry>()

  function settle(entry: PendingEntry, decision: ConnectorApprovalDecision): boolean {
    if (entry.settled) return false
    entry.settled = true
    if (entry.timer) clearTimeout(entry.timer)
    pending.delete(entry.token)
    entry.resolve(decision)
    return true
  }

  return {
    create(platform, threadKey) {
      const token = `ca_${randomUUID().slice(0, 8)}`
      let resolveFn: (decision: ConnectorApprovalDecision) => void = () => {}
      const promise = new Promise<ConnectorApprovalDecision>((resolve) => { resolveFn = resolve })
      const entry: PendingEntry = { token, platform, threadKey, createdAt: Date.now(), resolve: resolveFn, timer: null, settled: false }
      if (timeoutMs > 0) {
        entry.timer = setTimeout(() => { settle(entry, 'deny') }, timeoutMs)
        // Don't keep the process alive solely for a parked approval.
        entry.timer?.unref?.()
      }
      pending.set(token, entry)
      return { token, promise }
    },
    resolve(token, decision) {
      const entry = pending.get(token)
      if (!entry) return false
      return settle(entry, decision)
    },
    resolveByThreadKey(threadKey, decision) {
      let newest: PendingEntry | null = null
      for (const entry of pending.values()) {
        if (entry.threadKey !== threadKey) continue
        if (!newest || entry.createdAt > newest.createdAt) newest = entry
      }
      if (!newest) return false
      return settle(newest, decision)
    },
    size() {
      return pending.size
    },
  }
}

// ---- Inline-control markup (pure builders) ----
// Each returns the platform-native payload fragment for an approve/deny pair. The token is
// embedded in the action id / callback_data so the inbound callback resolves the exact request.

/** Slack Block Kit: a context section + an actions block with two buttons. */
export function slackApprovalBlocks(request: ChannelApprovalRequest, token: string): unknown[] {
  const text = [`*Approval needed:* ${request.title}`, request.summary, request.command ? `\`${request.command}\`` : '']
    .filter(Boolean)
    .join('\n')
  return [
    { type: 'section', text: { type: 'mrkdwn', text } },
    {
      type: 'actions',
      block_id: `vontra_approval:${token}`,
      elements: [
        { type: 'button', style: 'primary', text: { type: 'plain_text', text: 'Approve' }, action_id: 'vontra_approve', value: `approve:${token}` },
        { type: 'button', style: 'danger', text: { type: 'plain_text', text: 'Deny' }, action_id: 'vontra_deny', value: `deny:${token}` },
      ],
    },
  ]
}

/** Discord message components: one action row with two buttons (custom_id carries the token). */
export function discordApprovalComponents(token: string): unknown[] {
  return [
    {
      type: 1, // action row
      components: [
        { type: 2, style: 3, label: 'Approve', custom_id: `vontra_approve:${token}` }, // style 3 = success/green
        { type: 2, style: 4, label: 'Deny', custom_id: `vontra_deny:${token}` }, // style 4 = danger/red
      ],
    },
  ]
}

/** Telegram inline keyboard: callback_data carries the decision + token. */
export function telegramApprovalKeyboard(token: string): { inline_keyboard: { text: string; callback_data: string }[][] } {
  return {
    inline_keyboard: [[
      { text: '✅ Approve', callback_data: `/approve:${token}` },
      { text: '❌ Deny', callback_data: `/deny:${token}` },
    ]],
  }
}
