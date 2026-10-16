// Pi engine (VST-033): drive the installed `pi` coding agent (@earendil-works/pi-coding-agent)
// as a first-class thread engine, mirroring claude-cli.ts. Wire facts taken from the INSTALLED
// pi 0.80.3 — its `--help`, the docs it ships (docs/rpc.md, docs/json.md), and a live rpc-mode
// probe on this machine — not from memory:
// - `pi --mode rpc` speaks JSONL over stdin/stdout: commands in ({"type":"prompt","message"},
//   {"type":"steer","message"}, ...), events + id-correlated `response` lines out.
// - Streaming arrives as `message_update` events whose `assistantMessageEvent` is
//   `{type:"text_delta",delta}` or `{type:"thinking_delta",delta}` (same shapes as json mode).
// - `steer` queues a mid-turn user message, delivered after the current assistant turn's tool
//   calls and before the next LLM call (docs/rpc.md) — the live-steering seam.
// - `agent_end` marks the whole agentic run complete; stdin EOF then exits the process
//   cleanly (probe: exit 0 within ms), so no kill is needed on the happy path.
// - `--session-id <id>` uses an exact project session id, CREATING IT IF MISSING — so we mint
//   our own uuid up front (probe-verified accepted) and resume can never fail "not found".
// - rpc framing is STRICT LF-delimited JSONL: node readline is explicitly called out as
//   non-compliant (it also splits on U+2028/U+2029, which are valid inside JSON strings), so
//   stdout is split manually on \n only.
// - `--thinking <level>`: off | minimal | low | medium | high | xhigh.

import { randomUUID } from 'node:crypto'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { chmod, mkdtemp, readdir, rm, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { assertPiCredentialHandoff } from './cli-credential-policy'
import { resolveCliBinary, isSafeArgvId } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { cliConfinement, planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { EngineSteeringChannel } from './steering'
import type { ApiMessage, ApprovalMode, Provider } from './types'

export { assertPiCredentialHandoff } from './cli-credential-policy'

export interface PiResumeCursor {
  sessionId: string
}

export type PiCliChunk =
  | { type: 'content'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'resume_cursor'; cursor: PiResumeCursor }

export interface PiCliParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  cwd?: string | null
  approvalMode?: ApprovalMode
  reasoningEffort?: string
  resumeCursor?: PiResumeCursor | null
  // Live mid-turn steering (VST-033): cues become rpc `steer` commands on the open stdin.
  steering?: EngineSteeringChannel
  // Test seams: session-id minting and the steer-ack timeout.
  newSessionId?: () => string
  steerAckTimeoutMs?: number
  signal?: AbortSignal
}

const SILENCE_TIMEOUT_MS = 180_000
const TOTAL_TIMEOUT_MS = 1_200_000
const STEER_ACK_TIMEOUT_MS = 2_000
const PI_SYSTEM_PROMPT_PREFIX = 'vontra-pi-system-'
const LEGACY_PROMPT_MAX_AGE_MS = 24 * 60 * 60 * 1000

export interface PiSystemPromptFile {
  path: string
  cleanup(): Promise<void>
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error: any) {
    return error?.code !== 'ESRCH'
  }
}

/** Remove private prompt directories abandoned by a crashed TensorFold process. */
export async function cleanupStalePiSystemPrompts(
  root = os.tmpdir(),
  options: { currentPid?: number; now?: number; isProcessAlive?: (pid: number) => boolean } = {},
): Promise<void> {
  const currentPid = options.currentPid ?? process.pid
  const now = options.now ?? Date.now()
  const alive = options.isProcessAlive ?? processIsAlive
  const entries = await readdir(root, { withFileTypes: true }).catch(() => [])
  await Promise.all(entries.map(async (entry) => {
    if (!entry.isDirectory() || !entry.name.startsWith(PI_SYSTEM_PROMPT_PREFIX)) return
    const match = /^vontra-pi-system-(\d+)-/.exec(entry.name)
    const dir = path.join(root, entry.name)
    if (match) {
      const ownerPid = Number(match[1])
      if (ownerPid === currentPid || alive(ownerPid)) return
      await rm(dir, { recursive: true, force: true }).catch(() => undefined)
      return
    }
    // Pre-PID directories cannot be attributed safely while fresh; remove only
    // clearly stale leftovers from the previous implementation.
    const info = await stat(dir).catch(() => null)
    if (info && now - info.mtimeMs > LEGACY_PROMPT_MAX_AGE_MS) {
      await rm(dir, { recursive: true, force: true }).catch(() => undefined)
    }
  }))
}

export async function createPiSystemPromptFile(
  text: string,
  root = os.tmpdir(),
  pid = process.pid,
): Promise<PiSystemPromptFile> {
  await cleanupStalePiSystemPrompts(root, { currentPid: pid })
  const dir = await mkdtemp(path.join(root, `${PI_SYSTEM_PROMPT_PREFIX}${pid}-`))
  try {
    await chmod(dir, 0o700)
    const file = path.join(dir, 'prompt.txt')
    await writeFile(file, text, { encoding: 'utf8', mode: 0o600, flag: 'wx' })
    await chmod(file, 0o600)
    let cleaned = false
    return {
      path: file,
      cleanup: async () => {
        if (cleaned) return
        cleaned = true
        await rm(dir, { recursive: true, force: true }).catch(() => undefined)
      },
    }
  } catch (error) {
    await rm(dir, { recursive: true, force: true }).catch(() => undefined)
    throw error
  }
}

function binary(provider: Provider): string {
  return resolveCliBinary(provider.cliPath, ['pi'])
}

// TensorFold reasoning efforts → pi's --thinking levels (verified set: off/minimal/low/medium/
// high/xhigh). 'max'/'ultrathink'-style efforts clamp to xhigh, pi's ceiling.
export function piThinkingLevel(value?: string): string {
  const v = (value || '').toLowerCase()
  if (v === 'off' || v === 'none') return 'off'
  if (v === 'minimal') return 'minimal'
  if (v === 'low') return 'low'
  if (v === 'high') return 'high'
  if (v === 'xhigh' || v === 'extra-high' || v === 'extra high' || v === 'max' || v === 'ultrathink' || v === 'ultracode') return 'xhigh'
  return 'medium'
}

// Read-only approval mode restricts pi to its read tool; other modes keep pi's defaults
// (pi has no sandbox flags — the tool allowlist is its only execution-narrowing control,
// so this is a UX guardrail, not a sandbox boundary).
export function piToolsForApproval(approvalMode?: ApprovalMode): string[] | null {
  if (approvalMode === 'readonly') return ['read']
  return null
}

// rpc-mode args. The session id is ALWAYS ours to pass: a resumed cursor reuses its id, a
// fresh turn mints one (create-if-missing semantics make both the same flag).
export function buildPiArgs(params: PiCliParams, systemPromptFile: string, sessionId: string, resuming: boolean): string[] {
  assertPiCredentialHandoff(params.provider)
  const args = ['--mode', 'rpc', '--thinking', piThinkingLevel(params.reasoningEffort)]
  if (params.model && params.model !== 'pi-default' && isSafeArgvId(params.model)) args.push('--model', params.model)
  const tools = piToolsForApproval(params.approvalMode)
  if (tools) args.push('--tools', tools.join(','))
  // The session id originates from pi's own output (a UUID) but is persisted and passed back
  // on resume — validate it can't masquerade as a CLI flag.
  if (isSafeArgvId(sessionId)) args.push('--session-id', sessionId)
  // A resumed session already carries the system prompt from the run that created it.
  if (!resuming && systemPromptFile) args.push('--append-system-prompt', systemPromptFile)
  return args
}

// Strict LF framing per docs/rpc.md: split ONLY on \n (tolerating a trailing \r), never on
// U+2028/U+2029 — those are valid inside JSON strings and node readline would corrupt them.
export function splitJsonlChunk(buffer: string, chunk: string): { lines: string[]; rest: string } {
  const combined = buffer + chunk
  const parts = combined.split('\n')
  const rest = parts.pop() ?? ''
  const lines = parts.map((line) => (line.endsWith('\r') ? line.slice(0, -1) : line)).filter((line) => line.trim().length > 0)
  return { lines, rest }
}

function compactStatusValue(value: string, max = 120): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

// Translate one pi JSONL event into a chunk, or null. Only delta text/thinking stream as
// content/reasoning; tool lifecycle becomes short status lines (tool NAME only, never args).
export function chunkFromPiEvent(event: any): PiCliChunk | null {
  const type = String(event?.type || '')
  if (type === 'message_update') {
    const inner = event.assistantMessageEvent || {}
    if (inner.type === 'text_delta' && typeof inner.delta === 'string' && inner.delta) return { type: 'content', text: inner.delta }
    if (inner.type === 'thinking_delta' && typeof inner.delta === 'string' && inner.delta) return { type: 'reasoning', text: inner.delta }
    return null
  }
  if (type === 'tool_execution_start') {
    return { type: 'reasoning', text: `Pi using ${compactStatusValue(String(event.toolName || 'tool'))}\n` }
  }
  if (type === 'tool_execution_end' && event.isError) {
    return { type: 'reasoning', text: `Pi tool ${compactStatusValue(String(event.toolName || 'tool'))} failed\n` }
  }
  if (type === 'auto_retry_start') {
    return { type: 'reasoning', text: `Pi retrying (attempt ${event.attempt ?? '?'})\n` }
  }
  return null
}

// Failure classification once the process exits: a rejected prompt command or a retry loop
// that gave up carries its own message; otherwise a nonzero exit surfaces stderr.
export function piFailureMessage(opts: { exitCode: number; stderr: string; retryFinalError?: string | null; promptRejected?: string | null }): string | null {
  if (opts.promptRejected?.trim()) return `Pi rejected the prompt: ${opts.promptRejected.trim().slice(0, 400)}`
  if (opts.retryFinalError?.trim()) return `Pi gave up after retries: ${opts.retryFinalError.trim().slice(0, 400)}`
  if (opts.exitCode === 0) return null
  const detail = opts.stderr.trim()
  return `Pi CLI exited with code ${opts.exitCode}${detail ? `: ${detail.slice(0, 400)}` : ''}`
}

// The newest user message — a resumed pi session holds the prior context server-side
// (well, session-file-side), so only the new input goes over (mirrors claude/codex).
export function latestPiUserBody(messages: ApiMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return contentText(messages[i].content).trim()
  }
  return ''
}

function splitMessages(messages: ApiMessage[]): { system: string; body: string } {
  const systemParts: string[] = []
  const bodyParts: string[] = []
  for (const message of messages) {
    const role = message.role || 'user'
    const content = contentText(message.content).trim()
    if (!content && !message.tool_calls?.length) continue
    if (role === 'system') systemParts.push(content)
    else if (role === 'assistant') bodyParts.push(`[assistant]\n${content}`)
    else if (role === 'tool') bodyParts.push(`[tool result: ${message.name || message.tool_call_id || 'tool'}]\n${content}`)
    else bodyParts.push(content)
  }
  return { system: systemParts.join('\n\n').trim(), body: bodyParts.join('\n\n').trim() }
}

export async function* streamPiCli(params: PiCliParams): AsyncGenerator<PiCliChunk> {
  const resume = params.resumeCursor?.sessionId ? params.resumeCursor : null
  const { system, body } = splitMessages(params.messages)
  const effectiveBody = resume ? (latestPiUserBody(params.messages) || body) : body
  if (!effectiveBody || params.signal?.aborted) return
  const sessionId = resume?.sessionId ?? (params.newSessionId ?? randomUUID)()
  assertPiCredentialHandoff(params.provider)

  // Pi accepts a file path for --append-system-prompt. Use a private temporary
  // file so the full system prompt never appears in process listings.
  const systemPrompt = !resume && system ? await createPiSystemPromptFile(system, cliConfinement(params.provider)?.tmpDir) : null
  const systemPromptFile = systemPrompt?.path ?? ''

  const bin = binary(params.provider)
  const args = buildPiArgs(params, systemPromptFile, sessionId, Boolean(resume))
  let proc: ChildProcessWithoutNullStreams
  try {
    const run = planningCommand(params.provider, bin, args, { cwd: params.cwd || undefined, env: cliSpawnEnv(params.provider.cliEnv) })
    proc = spawn(run.command, run.args, {
      cwd: run.cwd,
      detached: detachedProcessGroup(),
      env: run.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
  } catch (error) {
    await systemPrompt?.cleanup()
    throw error
  }

  try {

  let spawnErrorMessage = ''
  let watchdogKilled = ''
  let lastOutputAt = Date.now()
  const startedAt = lastOutputAt
  const watchdog = setInterval(() => {
    if (proc.exitCode != null) return
    const now = Date.now()
    if (now - startedAt > TOTAL_TIMEOUT_MS) watchdogKilled = 'Pi CLI total timeout exceeded.'
    else if (now - lastOutputAt > SILENCE_TIMEOUT_MS) watchdogKilled = `Pi CLI was silent for ${Math.round(SILENCE_TIMEOUT_MS / 1000)}s.`
    if (watchdogKilled) terminateChildProcess(proc, { processGroup: true })
  }, 5_000)
  proc.once('error', (e) => { spawnErrorMessage = e.message })

  const abort = () => {
    if (proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }
  params.signal?.addEventListener('abort', abort, { once: true })

  let stderr = ''
  proc.stderr.on('data', (d) => {
    stderr = (stderr + String(d)).slice(-4000)
  })

  // --- rpc command plumbing -------------------------------------------------------------
  let stdinClosed = false
  const writeCommand = (command: Record<string, unknown>): boolean => {
    if (stdinClosed || proc.exitCode != null) return false
    try {
      proc.stdin.write(JSON.stringify(command) + '\n')
      return true
    } catch {
      return false
    }
  }
  const endStdin = () => {
    if (stdinClosed) return
    stdinClosed = true
    try { proc.stdin.end() } catch { /* spawn failure path reports below */ }
  }

  // Steer acks: `steer` responses are id-correlated, so acceptance is the server's word,
  // not an optimistic write. A missing ack within the timeout counts as refused.
  let steerSeq = 0
  let agentEnded = false
  const pendingSteers = new Map<string, (accepted: boolean) => void>()
  const disconnectSteering = params.steering?.connect(
    (input) => new Promise<boolean>((resolve) => {
      const ackId = `steer-${++steerSeq}`
      if (!writeCommand({ id: ackId, type: 'steer', message: input.text })) return resolve(false)
      const timer = setTimeout(() => {
        pendingSteers.delete(ackId)
        resolve(false)
      }, params.steerAckTimeoutMs ?? STEER_ACK_TIMEOUT_MS)
      pendingSteers.set(ackId, (accepted) => {
        clearTimeout(timer)
        pendingSteers.delete(ackId)
        resolve(accepted)
      })
    }),
    () => !stdinClosed && !agentEnded && proc.exitCode == null,
  )

  writeCommand({ id: 'prompt-1', type: 'prompt', message: effectiveBody })

  let retryFinalError: string | null = null
  let promptRejected: string | null = null
  let streamed = false
  let sawAgentEnd = false

  // Manual LF-only framing (see header): a queue of parsed events drained by the generator.
  let lineBuffer = ''
  const eventQueue: any[] = []
  let wakeQueue: (() => void) | null = null
  let stdoutDone = false
  // Steer acks settle HERE, in the data handler, not in the generator's consume loop — an
  // ack must never wait on downstream chunk consumption (the offering pump would deadlock
  // against a paused consumer otherwise).
  const admitEvent = (event: any) => {
    if (event?.type === 'response') {
      const id = typeof event.id === 'string' ? event.id : ''
      const settle = pendingSteers.get(id)
      if (settle) {
        settle(event.success === true)
        return
      }
    }
    eventQueue.push(event)
  }
  proc.stdout.on('data', (d) => {
    const { lines, rest } = splitJsonlChunk(lineBuffer, String(d))
    lineBuffer = rest
    lastOutputAt = Date.now()
    for (const raw of lines) {
      try { admitEvent(JSON.parse(raw)) } catch { /* skip malformed line */ }
    }
    wakeQueue?.()
  })
  proc.stdout.on('end', () => {
    if (lineBuffer.trim()) {
      try { admitEvent(JSON.parse(lineBuffer)) } catch { /* torn tail */ }
      lineBuffer = ''
    }
    stdoutDone = true
    wakeQueue?.()
  })

  const nextEvent = async (): Promise<any | null> => {
    while (true) {
      if (eventQueue.length) return eventQueue.shift()
      if (stdoutDone || params.signal?.aborted) return null
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 250)
        wakeQueue = () => { clearTimeout(timer); resolve() }
      })
      wakeQueue = null
    }
  }

  try {
    while (true) {
      const event = await nextEvent()
      if (event == null) break
      const type = String(event?.type || '')
      if (type === 'response') {
        if (event.command === 'prompt' && event.success === false) {
          promptRejected = String(event.error || event.message || 'prompt refused')
          endStdin()
        }
        continue
      }
      if (type === 'auto_retry_end' && !event.success && typeof event.finalError === 'string') retryFinalError = event.finalError
      if (type === 'agent_end') {
        sawAgentEnd = true
        agentEnded = true
        // Probe-verified: stdin EOF after agent_end exits pi cleanly (code 0) — no kill.
        endStdin()
        continue
      }
      const chunk = chunkFromPiEvent(event)
      if (chunk) {
        if (chunk.type === 'content') streamed = true
        yield chunk
      }
    }
  } finally {
    disconnectSteering?.()
    for (const settle of pendingSteers.values()) settle(false)
    pendingSteers.clear()
    endStdin()
    clearInterval(watchdog)
    params.signal?.removeEventListener('abort', abort)
    // Consumer stopped iterating early (generator .return()): kill rather than orphan.
    if (!stdoutDone && proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }

  const exitCode: number = await new Promise((resolve) => {
    if (proc.exitCode != null) resolve(proc.exitCode)
    else proc.once('close', (code) => resolve(code ?? 0))
  })
  if (params.signal?.aborted) return
  if (spawnErrorMessage) throw new Error(`Pi CLI not found at ${bin}: ${spawnErrorMessage}`)
  if (watchdogKilled) throw new Error(watchdogKilled)
  const failure = piFailureMessage({ exitCode, stderr, retryFinalError, promptRejected })
  if (failure) throw new Error(failure)
  if (!sawAgentEnd || !streamed) throw new Error('Pi CLI finished without a final message.')
  yield { type: 'resume_cursor', cursor: { sessionId } }
  } finally {
    await systemPrompt?.cleanup()
  }
}
