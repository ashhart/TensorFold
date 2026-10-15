// Run: jiti electron/agent/approval-risk.test.ts
//
// Heuristic risk-band approval (VST-048). Codex's Guardian auto-approves by risk
// level via a model call; TensorFold's approval is static user-set rules. This module
// is the deterministic analog: it maps the already-computed commandClass /
// connectorActionClass taxonomies (see approval-policy.ts) into a 4-level risk
// band that the approval gate uses to decide auto-allow vs prompt in Full Auto.
//
// Zero model cost, fully auditable, matches TensorFold's security-first posture.
// The risk reason is surfaced in the timeline so the user sees WHY something
// auto-ran. A model-assessed reviewer is deliberately deferred —
// the seam is runDelegateChild/oracle; this is the conservative baseline.

export type RiskBand = 'low' | 'medium' | 'high' | 'critical'

export interface RiskAssessment {
  band: RiskBand
  /** Short, human-readable reason shown in the timeline (e.g. "low risk: read-only"). */
  reason: string
}

// Command classes -> risk band. Mirrors the commandClass taxonomy in
// approval-policy.ts. The mapping is conservative: anything destructive or
// secret-touching is critical; network/package/git-mutate is high; network-fetch
// and git-read are medium (mutating-ish or externally observable); the safe
// read/build/test trio is low.
const COMMAND_CLASS_RISK: Record<string, RiskBand> = {
  'empty-command': 'low',
  'read-only': 'low',
  'test': 'low',
  'build': 'low',
  'git-read': 'medium',
  'network-fetch': 'medium',
  'shell': 'medium',
  'git-mutate': 'high',
  'network-shell': 'high',
  'package-install': 'high',
  'compound-shell': 'high',
  'destructive-filesystem': 'critical',
  'secret-read': 'critical',
  'system-control': 'critical',
}

// Connector action classes -> risk band. Mirrors connectorActionClass.
// Outbound/irreversible actions are critical (payments, deletes, permission
// changes) — these reuse the requiresFreshConnectorApproval "always re-confirm"
// semantics even with a saved rule. Sends/replies/browser-submit are high.
// Drafts and reads are low.
const CONNECTOR_CLASS_RISK: Record<string, RiskBand> = {
  'read': 'low',
  'draft': 'low',
  'approval-resolution': 'medium',
  'calendar-create': 'medium',
  'calendar-update': 'high',
  'reply': 'high',
  'send': 'high',
  'browser-submit': 'high',
  'unknown': 'high',
  'delete': 'critical',
  'account-permission-change': 'critical',
  'payment-purchase': 'critical',
}

// Tools that act outside the sandbox (other apps' MCP servers, the desktop, a real browser).
const FULL_ACCESS_ASK_TOOLS = new Set(['controller_action', 'computer_use', 'browser_automation'])

/**
 * Full access runs what stays inside the sandbox without asking. Commands are pre-screened by
 * toolApprovalReasonForMode (deletes, secrets, system control, remote shells and git push still
 * carry a reason and ask); connectors run only their low-risk reads and drafts; MCP, desktop and
 * browser control always ask.
 */
export function fullAccessAutoAllows(kind: 'command' | 'connector' | 'tool', toolName: string, band: RiskBand): boolean {
  if (kind === 'command') return true
  if (kind === 'connector') return band === 'low'
  return !toolName.startsWith('mcp__') && !FULL_ACCESS_ASK_TOOLS.has(toolName)
}

const DEFAULT_COMMAND_RISK: RiskBand = 'medium'
const DEFAULT_CONNECTOR_RISK: RiskBand = 'high' // connectors touch external state — fail closed

function bandReason(band: RiskBand, klass: string): string {
  const labels: Record<RiskBand, string> = {
    low: 'low risk',
    medium: 'medium risk',
    high: 'high risk',
    critical: 'critical risk',
  }
  return `${labels[band]}: ${klass.replace(/-/g, ' ')}`
}

/**
 * Classify a tool call into a risk band from its already-computed classification.
 * For run_command, pass the commandClass. For connector tools, pass the
 * connectorActionClass. For everything else (file edits, MCP, etc.) pass the
 * tool name or 'unknown' and the band falls back to medium for mutating tools.
 *
 * Pure + deterministic. No network, no model call, no side effects.
 */
export function riskBandForClassification(classification: string | null | undefined, kind: 'command' | 'connector' | 'tool'): RiskAssessment {
  const klass = (classification ?? 'unknown').trim() || 'unknown'
  if (kind === 'command') {
    const band = COMMAND_CLASS_RISK[klass] ?? DEFAULT_COMMAND_RISK
    return { band, reason: bandReason(band, klass) }
  }
  if (kind === 'connector') {
    const band = CONNECTOR_CLASS_RISK[klass] ?? DEFAULT_CONNECTOR_RISK
    return { band, reason: bandReason(band, klass) }
  }
  // Generic mutating tool (write_file, edit_file, apply_patch, mcp__*, etc.).
  // File edits are medium (reversible via git/checkpoints); opaque tools (MCP,
  // unknown) are high because we can't see inside them.
  const band: RiskBand = klass === 'unknown' || klass.startsWith('mcp') ? 'high' : 'medium'
  return { band, reason: bandReason(band, klass === 'unknown' ? 'tool' : klass) }
}

/**
 * Should this risk band auto-allow in Full Auto? Only low-risk ever auto-runs
 * without a saved rule. Medium auto-runs only if a saved session/always rule
 * matches (the loop checks the rule separately). High and critical always prompt.
 *
 * `fullAutoRiskBand` is the user-configured ceiling (default 'low'): raising it
 * to 'medium' lets medium-risk auto-allow with a saved rule. High/critical are
 * never auto-allowed.
 */
export function shouldAutoAllowInFullAuto(band: RiskBand, fullAutoRiskBand: RiskBand = 'low', hasSavedRule: boolean): boolean {
  const ORDER: RiskBand[] = ['low', 'medium', 'high', 'critical']
  const ceiling = ORDER.indexOf(fullAutoRiskBand)
  const level = ORDER.indexOf(band)
  if (level > ceiling) return false
  if (band === 'low') return true
  if (band === 'medium') return hasSavedRule // medium needs a saved rule even at the higher ceiling
  return false // high/critical never auto-allow
}
