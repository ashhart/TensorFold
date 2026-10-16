// Host-side daily digest composer. The renderer already has away-brief.ts + daily-digest.ts,
// but the automation SCHEDULER runs in the main process, and the renderer↔main boundary forbids
// the host from importing renderer runtime (CLAUDE.md). So a scheduled 'digest' automation needs
// its own lean, boundary-clean composer over the host store's own shapes — that's this module.
//
// Pure and duck-typed: it reads only the handful of fields it needs, so a real Job/Mission/Room/
// ConnectorEvent satisfies the input structurally and it stays trivially unit-testable.

const DAY_MS = 24 * 3_600_000

// Fields are widened to `| null` so the real (richer, sometimes-nullable) Job/Mission/Room/
// ConnectorEvent types satisfy these structurally without a cast. All reads below guard nulls.
export interface DigestJobLike { kind?: string | null; status?: string | null; updatedAt?: number | null; finishedAt?: number | null; label?: string | null; summary?: string | null; owner?: string | null }
export interface DigestMissionLike { status?: string | null; name?: string | null; title?: string | null; updatedAt?: number | null }
export interface DigestRoomLike { name?: string | null; title?: string | null; updatedAt?: number | null }
export interface DigestConnectorEventLike { direction?: string | null; requiresApproval?: boolean | null; approvalStatus?: string | null; title?: string | null; createdAt?: number | null }

export interface HostDigestInput {
  now: number
  sinceMs?: number // window length; default 24h
  jobs?: DigestJobLike[]
  missions?: DigestMissionLike[]
  rooms?: DigestRoomLike[]
  connectorEvents?: DigestConnectorEventLike[]
  title?: string
  dateLabel?: string
}

const MISSION_ATTENTION = new Set(['waiting_approval', 'stalled', 'failed'])
const APPROVAL_RESOLVED = new Set(['approved', 'denied', 'sent', 'cancelled', 'expired'])

function dateLabelFor(now: number): string {
  try { return new Date(now).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' }) }
  catch { return new Date(now).toISOString().slice(0, 10) }
}
const ts = (v?: number | null) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

/**
 * Build a deterministic markdown briefing from recent host activity within a window
 * (default 24h): missions/jobs/connector-approvals needing action under "Needs attention",
 * completed jobs / updated rooms / inbound-message counts under "Updates". Pure.
 */
export function buildHostDigest(input: HostDigestInput): string {
  const now = input.now
  const since = now - (input.sinceMs ?? DAY_MS)
  const title = input.title || 'Daily brief'
  const dateLabel = input.dateLabel || dateLabelFor(now)
  const header = `# ${title} — ${dateLabel}`

  const attention: string[] = []
  const updates: string[] = []

  for (const m of input.missions ?? []) {
    if (m.status && MISSION_ATTENTION.has(m.status)) attention.push(`- [mission] ${m.name || m.title || 'Mission'} — ${m.status.replace(/_/g, ' ')}`)
  }
  for (const j of input.jobs ?? []) {
    const at = ts(j.finishedAt ?? j.updatedAt)
    if (at < since) continue
    const name = j.label || j.summary || j.owner || j.kind || 'job'
    if (j.status === 'failed') attention.push(`- [${j.kind || 'job'}] ${name} — failed`)
    else if (j.status === 'completed') updates.push(`- [${j.kind || 'job'}] ${name} — completed`)
  }
  for (const r of input.rooms ?? []) {
    if (ts(r.updatedAt) >= since) updates.push(`- [room] ${r.name || r.title || 'Room'} — updated`)
  }
  let inboundCount = 0
  for (const e of input.connectorEvents ?? []) {
    if (ts(e.createdAt) < since) continue
    if (e.requiresApproval && !(e.approvalStatus && APPROVAL_RESOLVED.has(e.approvalStatus))) {
      attention.push(`- [approval] ${e.title || 'Connector action needs approval'}`)
    } else if (e.direction === 'inbound') {
      inboundCount++
    }
  }
  if (inboundCount > 0) updates.push(`- ${inboundCount} new inbound message${inboundCount === 1 ? '' : 's'}`)

  if (!attention.length && !updates.length) return `${header}\n\nNo activity to report.`

  const headlineParts: string[] = []
  if (attention.length) headlineParts.push(`${attention.length} need you`)
  if (updates.length) headlineParts.push(`${updates.length} update${updates.length === 1 ? '' : 's'}`)
  const headline = headlineParts.join(' · ')

  const sections: string[] = []
  if (attention.length) sections.push(`## ⚠ Needs attention\n${attention.join('\n')}`)
  if (updates.length) sections.push(`## Updates\n${updates.join('\n')}`)

  return `${header}\n\n${headline}\n\n${sections.join('\n\n')}`
}
