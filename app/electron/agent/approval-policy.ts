import { createHash } from 'node:crypto'
import { commandPolicyText } from './command-policy'
import { webFetchHost } from './web-fetch-vetting'

export interface ApprovalPolicyScope {
  scope: 'tool' | 'connector'
  key: string
  label: string
  explanation: string
  classification?: string
}

export type CapabilityPolicyOutcome =
  | 'deny'
  | 'ask'
  | 'allow_read_only'
  | 'allow_low_risk_mutation'
  | 'require_reviewer'
  | 'require_user'
  | 'allow_once'
  | 'allow_session'
  | 'allow_always'

export interface CapabilityPolicyScopeFields {
  workspace?: string | null
  projectId?: string | null
  missionId?: string | null
  roomId?: string | null
  tool?: string | null
  capability?: string | null
  commandClass?: string | null
  pathGlob?: string | null
  connector?: string | null
  recipientOrPerson?: string | null
  modelRole?: string | null
  riskScore?: number | null
  duration?: 'once' | 'session' | 'always' | null
  budget?: string | number | null
  source?: string | null
}

export interface CapabilityPolicy {
  id: string
  scope: CapabilityPolicyScopeFields
  outcome: CapabilityPolicyOutcome
  label: string
  explanation: string
  createdAt: number
  updatedAt: number
}

export const CAPABILITY_POLICY_OUTCOMES: CapabilityPolicyOutcome[] = [
  'deny',
  'ask',
  'allow_read_only',
  'allow_low_risk_mutation',
  'require_reviewer',
  'require_user',
  'allow_once',
  'allow_session',
  'allow_always',
]

export function capabilityPolicyScopeKey(scope: CapabilityPolicyScopeFields): string {
  return shortHash(scope)
}

export function approvalPolicyScope(scope: 'tool' | 'connector', toolOrAction: string, args: unknown = {}): ApprovalPolicyScope {
  const key = String(toolOrAction || 'unknown').trim() || 'unknown'
  if (scope === 'connector') return connectorPolicyScope(key, normalizeArgs(args))

  const normalized = normalizeArgs(args)
  if (key === 'run_command') return commandPolicyScope(key, normalized)
  if (key === 'write_file' || key === 'edit_file' || key === 'apply_patch') return pathPolicyScope(key, normalized)
  if (key === 'email_compose') return emailPolicyScope(key, normalized)
  if (key === 'browser_automation') return browserAutomationPolicyScope(key, normalized)
  if (key === 'controller_action') return controllerActionPolicyScope(key, normalized)
  if (key === 'computer_use') return computerUsePolicyScope(key, normalized)
  if (key === 'launch_subagent') return subAgentPolicyScope(key, normalized)
  if (key === 'web_fetch') return webFetchPolicyScope(key, normalized)
  if (key.startsWith('mcp__')) return hashedPolicyScope(key, normalized, 'MCP tool call')
  return hashedPolicyScope(key, normalized, 'Tool call')
}

export function requiresSavedApprovalPolicy(policy: ApprovalPolicyScope): boolean {
  return policy.scope === 'tool' && policy.classification === 'package-install'
}

const FRESH_CONNECTOR_APPROVAL_CLASSES = new Set([
  'send',
  'reply',
  'calendar-create',
  'calendar-update',
  'payment-purchase',
  'account-permission-change',
  'browser-submit',
  'delete',
])

/** Durable allow rules never silently replay decision-sensitive external mutations. */
export function requiresFreshConnectorApproval(policy: ApprovalPolicyScope): boolean {
  return policy.scope === 'connector' && FRESH_CONNECTOR_APPROVAL_CLASSES.has(policy.classification ?? '')
}

function connectorPolicyScope(actionKey: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const parsed = parseConnectorAction(actionKey, args)
  const klass = connectorActionClass(parsed.connectorId, parsed.action)
  const target = connectorTarget(args)
  const hash = shortHash({ connectorId: parsed.connectorId, action: parsed.action, klass, target })
  const label = `${connectorName(parsed.connectorId)} ${connectorClassLabel(klass)}${target ? ` · ${target.slice(0, 48)}` : ''}`
  return {
    scope: 'connector',
    key: `${parsed.connectorId}:${klass}:${hash}`,
    label,
    classification: klass,
    explanation: `Allows ${connectorName(parsed.connectorId)} ${connectorClassLabel(klass)}${target ? ` for ${target}` : ''}, not other connector actions.`,
  }
}

function parseConnectorAction(actionKey: string, args: Record<string, unknown>): { connectorId: string; action: string } {
  const connectorId = cleanString(args.connectorId)
  const action = cleanString(args.action)
  if (connectorId || action) return { connectorId: connectorId || 'connector', action: action || actionKey }
  const [rawConnector, ...rest] = actionKey.split(':')
  return { connectorId: rawConnector || 'connector', action: rest.join(':') || rawConnector || 'unknown' }
}

function connectorTarget(args: Record<string, unknown>): string {
  return cleanString(args.recipient)
    || cleanString(args.to)
    || cleanString(args.person)
    || cleanString(args.actor)
    || cleanString(args.target)
    || cleanString(args.threadKey)
    || cleanString(args.account)
    || cleanString(args.url)
}

function connectorActionClass(connectorId: string, action: string): string {
  const text = `${connectorId}:${action}`.toLowerCase().replace(/[^a-z0-9]+/g, '_')
  if (/(payment|purchase|checkout|invoice_pay|subscribe|billing)/.test(text)) return 'payment-purchase'
  if (/(permission|role|rbac|invite|member|access|account|oauth|token|credential)/.test(text)) return 'account-permission-change'
  if (/(browser|form).*submit|submit.*(form|browser)|post_form/.test(text)) return 'browser-submit'
  if (/calendar.*update|update.*calendar|event_update|reschedule/.test(text)) return 'calendar-update'
  if (/calendar.*create|create.*calendar|event_create|schedule_meeting/.test(text)) return 'calendar-create'
  if (/(delete|remove|archive|trash|discard)/.test(text)) return 'delete'
  if (/(send|send_requested|deliver|publish|post_message|message_send)/.test(text)) return 'send'
  if (/(reply|respond|message_compose|comment)/.test(text)) return 'reply'
  if (/(draft|compose|proposal|suggest)/.test(text)) return 'draft'
  if (/(approval_approved|approval_denied|approval_resolved)/.test(text)) return 'approval-resolution'
  if (/(message_received|document_indexed|email_sync|webhook_test|window_captured|summary|list|read)/.test(text)) return 'read'
  return 'unknown'
}

function connectorClassLabel(klass: string): string {
  const labels: Record<string, string> = {
    'draft': 'draft',
    'send': 'send',
    'reply': 'reply',
    'delete': 'delete',
    'calendar-create': 'calendar create',
    'calendar-update': 'calendar update',
    'browser-submit': 'browser submit',
    'payment-purchase': 'payment or purchase',
    'account-permission-change': 'account or permission change',
    'approval-resolution': 'approval resolution',
    'read': 'read action',
    'unknown': 'connector action',
  }
  return labels[klass] ?? klass.replace(/-/g, ' ')
}

function connectorName(connectorId: string): string {
  const names: Record<string, string> = {
    email: 'Email',
    slack: 'Slack',
    whatsapp: 'WhatsApp',
    imessage: 'iMessage',
    telegram: 'Telegram',
    docs: 'Documents',
    calendar: 'Calendar',
    controller: 'Controller',
    webhook: 'Webhook',
    browser: 'Browser',
  }
  return names[connectorId] ?? connectorId
}

function commandPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const command = cleanString(args.command)
  const cwd = cleanString(args.cwd)
  const klass = commandClass(command)
  const hash = shortHash({ command, cwd, klass })
  const label = cwd ? `${klass} in ${cwd}` : klass
  return {
    scope: 'tool',
    key: `${tool}:${klass}:${hash}`,
    label,
    classification: klass,
    explanation: `Allows this ${klass} command signature${cwd ? ` in ${cwd}` : ''}, not arbitrary shell commands.`,
  }
}

// apply_patch carries its target file(s) inside args.patch (`*** Update/Add/Delete File:`
// headers) or args.replacements[].path — NOT args.path. Pull them out so each patch scopes
// to its real files instead of collapsing to one shared key.
function applyPatchTargets(args: Record<string, unknown>): string[] {
  const paths = new Set<string>()
  // The executor (replacementHunksFromArgs) falls back to the TOP-LEVEL args.path when a
  // replacement omits its own path, and the single-replacement shape ({path,old_string,
  // new_string}) writes to args.path directly — so args.path is a real target and must be
  // part of the key, or different-file patches would collapse to one shared key.
  const topPath = cleanString(args.path)
  const replacements = (args as { replacements?: unknown }).replacements
  if (Array.isArray(replacements)) {
    for (const r of replacements) {
      const p = cleanString((r as { path?: unknown })?.path) || topPath
      if (p) paths.add(p)
    }
  } else if (topPath) {
    paths.add(topPath)
  }
  const patch = typeof args.patch === 'string' ? args.patch : ''
  for (const m of patch.matchAll(/\*\*\*\s+(?:Update|Add|Delete) File:\s*(.+)/gi)) {
    const p = m[1].trim()
    if (p) paths.add(p)
  }
  return [...paths].sort()
}

function pathPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  if (tool === 'apply_patch') {
    const cwd = cleanString(args.cwd)
    const targets = applyPatchTargets(args)
    if (targets.length) {
      const label = targets.join(', ')
      return {
        scope: 'tool',
        key: `${tool}:paths:${shortHash({ targets, cwd })}`,
        label: `${tool} ${label}`,
        explanation: `Allows ${tool} for ${label}${cwd ? ` under ${cwd}` : ''}, not every ${tool} call.`,
      }
    }
    // Fail closed: no target extractable -> scope to THIS exact patch payload (incl. the
    // path/old/new fields) so an "Always" can never widen to other files or other edits.
    const replacements = (args as { replacements?: unknown }).replacements
    return {
      scope: 'tool',
      key: `${tool}:patch:${shortHash({ patch: typeof args.patch === 'string' ? args.patch : '', replacements: replacements ?? null, path: cleanString(args.path), old_string: cleanString(args.old_string), new_string: cleanString(args.new_string), cwd })}`,
      label: `${tool} (this patch only)`,
      explanation: `Allows only this exact ${tool} payload, not every ${tool} call.`,
    }
  }
  const path = cleanString(args.path) || cleanString(args.file) || cleanString(args.target) || '<no-path>'
  const cwd = cleanString(args.cwd)
  return {
    scope: 'tool',
    key: `${tool}:path:${shortHash({ path, cwd })}`,
    label: `${tool} ${path}`,
    explanation: `Allows ${tool} for ${path}${cwd ? ` under ${cwd}` : ''}, not every ${tool} call.`,
  }
}

// Tools whose effects reach OUTSIDE the (checkpointed) workspace — sending messages/email,
// controlling the Mac, driving the browser, mutating the calendar. Full Auto must prompt for
// these even though it may auto-run in-workspace edits/reads. run_command is handled
// separately by commandApprovalReason.
export function externalEffectApprovalReason(tool: string, args: Record<string, unknown> = {}): string | null {
  // MCP tools run arbitrary server-defined side effects (post a message, hit an API, drive
  // the computer, schedule a task). The server's own read-only hints aren't trusted
  // (isReadOnlyTool), so Full Auto must never blanket-allow them — always require approval.
  if (tool.startsWith('mcp__')) return 'this MCP tool may have external side effects and needs approval'
  switch (tool) {
    case 'tool_call':
      return 'calling a deferred tool may have external side effects and needs approval'
    case 'controller_action':
      return 'controlling apps on this Mac requires approval'
    case 'computer_use':
      return 'operating another app on this Mac requires approval'
    case 'browser_automation': {
      const klass = browserAutomationClass(cleanString(args.action) || 'automation')
      // Inspection/screenshot are read-ish; interaction/navigate/tab change external state.
      return klass === 'browser-inspection' ? null : 'browser actions that change external state require approval'
    }
    case 'message_compose':
      return 'sending a message requires approval'
    case 'email_compose':
      // Drafting is harmless (draft-first design); only an actual send is external.
      return cleanString(args.mode) === 'send' ? 'sending email requires approval' : null
    case 'calendar_create':
    case 'calendar_update':
      return 'changing the calendar requires approval'
    default:
      return null
  }
}

function emailPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const mode = cleanString(args.mode) || 'draft'
  const to = cleanString(args.to) || '<recipient>'
  const subject = cleanString(args.subject) || '<subject>'
  return {
    scope: 'tool',
    key: `${tool}:${mode}:${shortHash({ to, subject })}`,
    label: `${mode} email to ${to}`,
    explanation: `Allows ${mode} email composition only for ${to} with this subject signature.`,
  }
}

function browserAutomationPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const action = cleanString(args.action) || 'automation'
  const target = cleanString(args.url)
    || cleanString(args.selector)
    || cleanString(args.ref)
    || cleanString(args.tabId)
    || '<current tab>'
  const classification = browserAutomationClass(action)
  return {
    scope: 'tool',
    key: `${tool}:${classification}:${shortHash({ action, target, sessionId: cleanString(args.sessionId) })}`,
    label: `Browser ${action} · ${target.slice(0, 48)}`,
    classification,
    explanation: `Allows Browser ${action} for ${target}, not arbitrary browser automation.`,
  }
}

function browserAutomationClass(action: string): string {
  const normalized = action.toLowerCase().replace(/[^a-z0-9]+/g, '_')
  if (normalized === 'navigate' || normalized === 'tab_new') return 'browser-navigate'
  if (normalized === 'click' || normalized === 'type' || normalized === 'fill_form') return 'browser-interaction'
  if (normalized === 'tab_switch' || normalized === 'tab_close') return 'browser-tab'
  if (normalized === 'screenshot' || normalized === 'console_logs' || normalized === 'network_errors' || normalized === 'accessibility_snapshot') return 'browser-inspection'
  return 'browser-automation'
}

function controllerActionPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const action = cleanString(args.action) || 'action'
  const appName = cleanString(args.appName) || '<app>'
  const target = cleanString(args.keystroke)
    || (Array.isArray(args.menuPath) ? args.menuPath.map(cleanString).filter(Boolean).join(' > ') : cleanString(args.menuPath))
    || appName
  const classification = controllerActionClass(action)
  return {
    scope: 'tool',
    key: `${tool}:${classification}:${shortHash({ action, appName, target })}`,
    label: `Controller ${action} · ${appName} · ${target.slice(0, 48)}`,
    classification,
    explanation: `Allows Controller ${action} for ${appName} (${target}), not arbitrary Mac app control.`,
  }
}

// One grant per app: approving Safari for the session covers every computer_use action in Safari.
function computerUsePolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const app = cleanString(args.app) || '<app>'
  return {
    scope: 'tool',
    key: `${tool}:computer-use:${shortHash({ app: app.toLowerCase() })}`,
    label: `Computer use · ${app}`,
    classification: 'computer-use',
    explanation: `Allows TensorFold to read and operate ${app} on this Mac, not other apps.`,
  }
}

function controllerActionClass(action: string): string {
  const normalized = action.toLowerCase().replace(/[^a-z0-9]+/g, '_')
  if (normalized === 'keystroke') return 'controller-keystroke'
  if (normalized === 'menu_item') return 'controller-menu'
  if (normalized === 'activate_app') return 'controller-activate'
  return 'controller-action'
}

function subAgentPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const task = cleanString(args.task) || cleanString(args.name) || '<task>'
  const providerId = cleanString(args.providerId)
  const model = cleanString(args.model)
  return {
    scope: 'tool',
    key: `${tool}:${shortHash({ task, providerId, model })}`,
    label: `sub-agent ${task.slice(0, 48)}`,
    explanation: `Allows launching this specific sub-agent task${model ? ` on ${model}` : ''}.`,
  }
}

// Scoped to the host, so "Always" vets one site rather than one exact URL.
function webFetchPolicyScope(tool: string, args: Record<string, unknown>): ApprovalPolicyScope {
  const host = webFetchHost(args)
  if (!host) return hashedPolicyScope(tool, args, 'Web fetch')
  return {
    scope: 'tool',
    key: `${tool}:host:${host}`,
    label: `Web fetch · ${host}`,
    explanation: `Allows web_fetch to ${host} only, not other sites.`,
    classification: 'network-read',
  }
}

function hashedPolicyScope(tool: string, args: Record<string, unknown>, label: string): ApprovalPolicyScope {
  return {
    scope: 'tool',
    key: `${tool}:${shortHash(args)}`,
    label: `${label}: ${tool}`,
    explanation: `Allows ${tool} only with this argument signature.`,
  }
}

export function commandClass(command: string): string {
  const rawText = command.trim()
  const text = commandPolicyText(rawText)
  const compact = text.toLowerCase()
  if (!text) return 'empty-command'
  if (isSystemControlCommand(compact)) return 'system-control'
  if (isSecretReadCommand(compact)) return 'secret-read'
  if (isDestructiveFilesystemCommand(compact)) return 'destructive-filesystem'
  if (isNetworkShellCommand(compact)) return 'network-shell'
  if (/(?:^|[\s;&|])(npm|pnpm|yarn|bun)\s+(i|install|add|dlx|exec|create)\b/.test(compact)
    || /(?:^|[\s;&|])(?:pip|pip3|python3?\s+-m\s+pip)\s+install\b/.test(compact)) return 'package-install'
  // Shell composition/redirection can hide a second command or turn an otherwise read-only
  // utility into a mutation. Classify it before the safe build/read allowlist so Full Auto asks.
  if (hasShellControlSyntax(rawText)) return 'compound-shell'
  if (/^\s*(npm|pnpm|yarn|bun)\s+(test|run\s+test|vitest|jest)\b/.test(text)) return 'test'
  if (/^\s*(npm|pnpm|yarn|bun)\s+(run\s+)?(build|typecheck|lint)\b/.test(text)) return 'build'
  if (/^\s*git\s+(status|diff|log|show|branch|rev-parse|ls-files)\b/.test(text)) return 'git-read'
  if (/^\s*git\s+(add|commit|push|checkout|switch|merge|rebase|reset|clean|worktree)\b/.test(text)) return 'git-mutate'
  if (/^\s*(curl|wget|http|https)\b/.test(text)) return 'network-fetch'
  if (isReadOnlyShellCommand(text)) return 'read-only'
  return 'shell'
}

// Includes the macOS automation tools: with TensorFold's Accessibility grant they could drive any app.
function isSystemControlCommand(text: string): boolean {
  return /(?:^|[\s;&|/])(sudo|launchctl|systemctl|service|shutdown|reboot|halt|pmset|killall|osascript|cliclick|shortcuts|automator)\b/.test(text)
}

function isSecretReadCommand(text: string): boolean {
  if (!/^\s*(cat|grep|rg|sed|awk|head|tail|less|more|find|ls|xxd|od|hexdump|strings|nl|tac|base64|openssl|dd|cp|mv|scp|rsync)\b/.test(text)) return false
  return /(^|[/. _=-])(\.env|env\.local|id_rsa|id_ed25519|credentials|credential|token|secret|passwd|shadow|\.npmrc|\.pypirc|\.netrc|aws_access_key|private[_-]?key)(\b|[/. _-])/i.test(text)
}

function isDestructiveFilesystemCommand(text: string): boolean {
  return /(?:^|[\s;&|])rm\s+[^;&|\n]*(?:-[a-z]*r[a-z]*\b|--recursive\b)/i.test(text)
    || /(?:^|[\s;&|])(chmod|chown|chgrp)\s+(?:-[a-z]*r[a-z]*\b|--recursive\b)/i.test(text)
    || /(?:^|[\s;&|])(mv|cp)\s+[^;&|\n]*\s+(\/|~\/)/i.test(text)
    || /(?:^|[\s;&|])git\s+reset\s+[^;&|\n]*--hard\b/i.test(text)
    || /(?:^|[\s;&|])git\s+clean\s+[^;&|\n]*-[^\s]*f/i.test(text)
}

function isNetworkShellCommand(text: string): boolean {
  return /(?:^|[\s;&|])(ssh|scp|sftp|rsync|nc|ncat|netcat|telnet|socat)\b/.test(text)
    || /(?:^|[\s;&|])(curl|wget)\b.*(\|\s*(sh|bash|zsh|python|perl|ruby)\b|>\s*\/tmp\/)/.test(text)
}

// Detect unquoted shell control syntax. Quoted `;`/`|` characters are data, while redirects,
// command substitutions, newlines, and actual separators make the command compound/opaque.
export function hasShellControlSyntax(command: string): boolean {
  let quote: "'" | '"' | null = null
  let escaped = false
  for (let i = 0; i < command.length; i++) {
    const ch = command[i]
    if (escaped) { escaped = false; continue }
    if (ch === '\\' && quote !== "'") { escaped = true; continue }
    if (quote) {
      if (ch === quote) quote = null
      continue
    }
    if (ch === "'" || ch === '"') { quote = ch; continue }
    if (ch === '`' || ch === '\n' || ch === ';' || ch === '|' || ch === '&' || ch === '>' || ch === '<') return true
    if (ch === '$' && command[i + 1] === '(') return true
  }
  return false
}

export function isReadOnlyShellCommand(command: string): boolean {
  const text = command.trim()
  if (!text || hasShellControlSyntax(text)) return false
  if (/\b(tee|xargs|sh|bash|zsh|python|node|perl|ruby)\b/i.test(text)) return false
  if (!/^(pwd|ls|cat|grep|rg|sed|awk|head|tail|less|more|find|wc|git\s+(status|diff|log|show|branch|rev-parse|ls-files))\b/i.test(text)) return false
  if (/^find\b/i.test(text) && /-(?:delete|exec|execdir|ok|okdir|fprintf|fls)\b/i.test(text)) return false
  return true
}

function normalizeArgs(input: unknown): Record<string, unknown> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return {}
  return stableValue(input) as Record<string, unknown>
}

function shortHash(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(stableValue(value))).digest('hex').slice(0, 16)
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue)
  if (!value || typeof value !== 'object') return typeof value === 'string' ? value.trim() : value
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== undefined)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => [key, stableValue(item)]))
}

function cleanString(value: unknown): string {
  return typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : ''
}
