import { spawn } from 'node:child_process'
import readline from 'node:readline'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { resolveCliBinary } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { ApiMessage, ApprovalMode, Provider } from './types'

export type CodexCliChunk =
  | { type: 'content'; text: string }
  | { type: 'reasoning'; text: string }

export interface CodexCliParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  cwd?: string | null
  approvalMode?: ApprovalMode
  reasoningEffort?: string
  serviceTier?: string
  signal?: AbortSignal
}

const AUTH_FAILURE = /(not\s+logged\s+in|please\s+(?:run\s+)?codex\s+login|unauthori[sz]ed|token\s+(?:expired|invalid)|authentication\s+(?:failed|required)|session\s+expired|re-?auth|credentials?\s+(?:invalid|expired|missing))/i
const SILENCE_TIMEOUT_MS = 180_000
const TOTAL_TIMEOUT_MS = 1_200_000

export function binary(provider: Provider): string {
  // Resolve to an absolute path so a GUI-launched app (stripped PATH) can still find the CLI.
  return resolveCliBinary(provider.cliPath, ['codex'])
}

function effort(value?: string): string {
  const v = (value || '').toLowerCase()
  if (v === 'minimal' || v === 'low') return 'low'
  if (v === 'high') return 'high'
  if (v === 'xhigh' || v === 'extra-high' || v === 'extra high' || v === 'max') return 'xhigh'
  return 'medium'
}

// TensorFold lists a couple of codex "models" that are really effort presets. codex only knows the real
// models gpt-5.5 / gpt-5.4 — "high"/"extra high" are reasoning-effort levels, NOT model ids — so
// `codex exec -m gpt-5.5-extra-high` 400s with "model is not supported". Map the aliases to a real
// model + effort. A plain model uses the separate reasoning-effort selector.
const CODEX_MODEL_ALIASES: Record<string, { model: string; effort: string }> = {
  'gpt-5.5-codex-high': { model: 'gpt-5.5', effort: 'high' },
  'gpt-5.5-extra-high': { model: 'gpt-5.5', effort: 'xhigh' },
}

export function codexModelAndEffort(model: string, reasoningEffort?: string): { model: string; effort: string } {
  const alias = CODEX_MODEL_ALIASES[(model || '').trim()]
  if (alias) return alias
  return { model, effort: effort(reasoningEffort) }
}

// codex reports turn failures as JSON events on STDOUT (type:"error" / "turn.failed"), not stderr —
// and nests the upstream API error as a JSON string. Pull out the human-readable message so the UI
// shows the real cause instead of a blank "exited with code 1".
function codexErrorMessage(event: any): string {
  const raw = event?.error?.message ?? event?.message
  if (typeof raw !== 'string' || !raw) return ''
  try { const inner = JSON.parse(raw); return inner?.error?.message || inner?.message || raw } catch { return raw }
}

export function sandbox(provider: Provider, approvalMode?: ApprovalMode): string {
  if (approvalMode === 'readonly') return 'read-only'
  // Full Auto suppresses TensorFold's safe-class prompts; it never disables the delegated CLI's
  // workspace boundary. Provider settings also cannot reintroduce danger-full-access in Auto.
  if (approvalMode === 'auto') return 'workspace-write'
  if (provider.cliSandbox) return provider.cliSandbox
  return 'workspace-write'
}

export function codexExecApprovalPolicy(provider: Provider, approvalMode?: ApprovalMode): string {
  // `codex exec` is non-interactive and cannot bridge command/file approval prompts back into
  // TensorFold. In exec mode, on-request approvals fail invisibly/awkwardly; use the sandbox as the
  // safety boundary and reserve approval bridging for a future app-server-backed session.
  if (approvalMode === 'auto' || approvalMode === 'readonly') return 'never'
  const policy = provider.cliApprovalPolicy || 'never'
  return policy === 'on-request' ? 'never' : policy
}

export function flattenMessages(messages: ApiMessage[]): string {
  const parts: string[] = []
  for (const message of messages) {
    const role = message.role || 'user'
    const content = contentText(message.content).trim()
    if (!content && !message.tool_calls?.length) continue
    if (role === 'system') parts.push(`[system]\n${content}`)
    else if (role === 'assistant') {
      const calls = message.tool_calls?.length
        ? `\n\n[assistant requested tools]\n${message.tool_calls.map((tc) => `${tc.function.name}: ${tc.function.arguments}`).join('\n')}`
        : ''
      parts.push(`[assistant]\n${content}${calls}`)
    } else if (role === 'tool') {
      parts.push(`[tool result: ${message.name || message.tool_call_id || 'tool'}]\n${content}`)
    } else {
      parts.push(content)
    }
  }
  return parts.join('\n\n').trim()
}

export function buildCodexArgs(params: CodexCliParams): string[] {
  const args = ['exec']
  const resolved = codexModelAndEffort(params.model, params.reasoningEffort)
  if (resolved.model && resolved.model !== 'codex-cli-default') args.push('-m', resolved.model)
  if (params.provider.cliProfile?.trim()) args.push('-p', params.provider.cliProfile.trim())
  args.push('-c', `model_reasoning_effort="${resolved.effort}"`)
  // `serviceTier` can come from persisted/user-edited settings. JSON string quoting is
  // compatible with TOML basic strings and prevents quotes, slashes, or control chars
  // from breaking the single `-c` override into malformed/injected TOML.
  if (params.serviceTier?.trim()) args.push('-c', `service_tier=${JSON.stringify(params.serviceTier.trim())}`)
  if (params.cwd) args.push('-C', params.cwd)
  args.push('-s', sandbox(params.provider, params.approvalMode))
  // `codex exec` (>= 0.125) dropped the top-level `--ask-for-approval` flag — it now exits 2 with
  // "unexpected argument '--ask-for-approval' found". Route the policy through the supported
  // config override; the sandbox stays active even when non-interactive Auto uses `never`.
  args.push('-c', `approval_policy="${codexExecApprovalPolicy(params.provider, params.approvalMode)}"`)
  args.push('--skip-git-repo-check', '--json', '-')
  return args
}

function textFromEvent(event: any, alreadyStreamed: boolean): string {
  const type = String(event?.type || '')
  if (type === 'response.output_text.delta' && typeof event.delta === 'string') return event.delta
  if (type.endsWith('.delta') && typeof event.delta === 'string') return event.delta
  if (type === 'item.completed') {
    const item = event.item || {}
    if (alreadyStreamed) return ''
    if (item.type === 'agent_message' && typeof item.text === 'string') return item.text
    if (typeof item.text === 'string') return item.text
    if (typeof item.content === 'string') return item.content
  }
  if (!alreadyStreamed && type === 'message' && typeof event.text === 'string') return event.text
  return ''
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

function compactStatusValue(value: string, max = 120): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

function codexItemStatus(item: any, phase: 'started' | 'completed'): string {
  const itemType = String(item?.type || item?.kind || '')
  const args = item?.args || item?.arguments || {}
  const command = compactStatusValue(firstString(item?.command, item?.cmd, args?.command, args?.cmd, item?.input))
  const tool = firstString(item?.name, item?.tool, item?.toolName, item?.function?.name, args?.name, args?.tool)
  if (itemType === 'agent_message') return ''
  if (itemType === 'reasoning') return phase === 'started' ? 'reasoning' : ''
  if (itemType === 'command_execution' || itemType === 'exec_command' || itemType === 'terminal_command') {
    return phase === 'started'
      ? command ? `running shell: ${command}` : 'running shell'
      : command ? `finished shell: ${command}` : 'finished shell'
  }
  if (itemType === 'tool_call' || itemType === 'function_call' || itemType === 'mcp_tool_call') {
    return phase === 'started'
      ? tool ? `using ${tool}` : 'using a tool'
      : tool ? `finished ${tool}` : 'finished a tool'
  }
  if (!itemType) return ''
  const label = itemType.replace(/_/g, ' ')
  return phase === 'started' ? label : `finished ${label}`
}

export function statusFromEvent(event: any): string {
  const type = String(event?.type || '')
  if (type === 'item.started') {
    const status = codexItemStatus(event?.item, 'started')
    if (status) return `Codex ${status}`
  }
  if (type === 'item.completed') {
    const status = codexItemStatus(event?.item, 'completed')
    if (status) return `Codex ${status}`
  }
  if (type === 'turn.started') return 'Codex turn started'
  if (type === 'turn.completed') return 'Codex turn completed'
  if (type === 'error' || type === 'turn.failed') {
    const message = compactStatusValue(codexErrorMessage(event), 160)
    if (message) return `Codex error: ${message}`
  }
  return ''
}

export async function checkCodexCli(provider: Provider): Promise<string> {
  const bin = binary(provider)
  return await new Promise((resolve, reject) => {
    const proc = spawn(bin, ['--version'], {
      detached: detachedProcessGroup(),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: cliSpawnEnv(provider.cliEnv),
    })
    let out = ''
    let err = ''
    const timeout = setTimeout(() => {
      terminateChildProcess(proc, { processGroup: true })
      reject(new Error('codex --version timed out'))
    }, 10_000)
    proc.stdout.on('data', (d) => { out += String(d) })
    proc.stderr.on('data', (d) => { err += String(d) })
    proc.on('error', (e) => {
      clearTimeout(timeout)
      reject(new Error(`codex CLI not found at ${bin}: ${e.message}`))
    })
    proc.on('close', (code) => {
      clearTimeout(timeout)
      if (code === 0) resolve(out.trim() || 'codex CLI ready')
      else reject(new Error((err || out || `codex --version exited ${code}`).trim()))
    })
  })
}

export async function* streamCodexCli(params: CodexCliParams): AsyncGenerator<CodexCliChunk> {
  const prompt = flattenMessages(params.messages)
  if (!prompt || params.signal?.aborted) return

  const bin = binary(params.provider)
  const run = planningCommand(params.provider, bin, buildCodexArgs(params), { cwd: params.cwd || undefined, env: cliSpawnEnv(params.provider.cliEnv) })
  const proc = spawn(run.command, run.args, {
    cwd: run.cwd,
    detached: detachedProcessGroup(),
    env: run.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  let spawnErrorMessage = ''
  let watchdogKilled = ''
  let lastOutputAt = Date.now()
  const startedAt = lastOutputAt
  const watchdog = setInterval(() => {
    if (proc.exitCode != null) return
    const now = Date.now()
    if (now - startedAt > TOTAL_TIMEOUT_MS) watchdogKilled = 'Codex CLI total timeout exceeded.'
    else if (now - lastOutputAt > SILENCE_TIMEOUT_MS) watchdogKilled = `Codex CLI was silent for ${Math.round(SILENCE_TIMEOUT_MS / 1000)}s.`
    if (watchdogKilled) terminateChildProcess(proc, { processGroup: true })
  }, 5_000)
  proc.once('error', (e) => { spawnErrorMessage = e.message })

  const abort = () => {
    if (proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }
  params.signal?.addEventListener('abort', abort, { once: true })

  let stderr = ''
  let codexError = ''
  proc.stderr.on('data', (d) => {
    stderr = (stderr + String(d)).slice(-4000)
  })

  try {
    proc.stdin.write(prompt)
    proc.stdin.end()
  } catch {
    // If spawn failed, the close/error path below produces the useful message.
  }

  const lines = readline.createInterface({ input: proc.stdout })
  let streamed = false
  let completed = false
  try {
    for await (const line of lines) {
      if (params.signal?.aborted) break
      const text = String(line || '').trim()
      if (!text) continue
      lastOutputAt = Date.now()
      let event: any
      try { event = JSON.parse(text) } catch { continue }
      const eventType = String(event?.type || '')
      if (eventType === 'error' || eventType === 'turn.failed') {
        const m = codexErrorMessage(event)
        if (m) codexError = m
      }
      const status = statusFromEvent(event)
      if (status) yield { type: 'reasoning', text: `${status}\n` }
      const content = textFromEvent(event, streamed)
      if (content) {
        streamed = true
        yield { type: 'content', text: content }
      }
    }
    completed = true // stdout reached EOF naturally; the exit-code await below is valid
  } finally {
    clearInterval(watchdog)
    params.signal?.removeEventListener('abort', abort)
    // If the consumer stopped iterating early (generator .return(), e.g. an upstream
    // cancel that didn't go through params.signal), the child is still running and the
    // exit-code await below never executes — kill it so `codex exec` isn't orphaned.
    if (!completed) lines.close()
    if (!completed && proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }

  const exitCode: number = await new Promise((resolve) => {
    if (proc.exitCode != null) resolve(proc.exitCode)
    else proc.once('close', (code) => resolve(code ?? 0))
  })
  if (params.signal?.aborted) return
  if (spawnErrorMessage) throw new Error(`Codex CLI not found at ${bin}: ${spawnErrorMessage}`)
  if (watchdogKilled) throw new Error(watchdogKilled)
  if (exitCode !== 0) {
    const detail = codexError || stderr
    if (detail && AUTH_FAILURE.test(detail)) throw new Error('Codex CLI is not authenticated. Run `codex login`, then retry this provider.')
    // Prefer codex's own JSON error (the real cause, e.g. "The 'X' model is not supported") over the
    // generic exit code — stderr is usually empty because codex writes errors to stdout.
    if (codexError) throw new Error(codexError)
    throw new Error(`Codex CLI exited with code ${exitCode}${stderr ? `: ${stderr.slice(0, 400)}` : ''}`)
  }
  if (!streamed) throw new Error('Codex CLI finished without a final message.')
}
