import { spawnCliProcess as spawn } from './cli-process'
import { chmod, mkdtemp, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { resolveCliBinary, isSafeArgvId } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { cliConfinement, planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { EngineSteeringChannel } from './steering'
import type { ApiMessage, ApprovalMode, Provider } from './types'

export interface ClaudeResumeCursor {
  sessionId: string
}

export type ClaudeCliChunk =
  | { type: 'content'; text: string }
  | { type: 'reasoning'; text: string }
  // Terminal chunk carrying the session id to persist onto the Thread for next time.
  | { type: 'resume_cursor'; cursor: ClaudeResumeCursor }

export interface ClaudeCliParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  cwd?: string | null
  approvalMode?: ApprovalMode
  reasoningEffort?: string

  resumeCursor?: ClaudeResumeCursor | null

  steering?: EngineSteeringChannel
  // Test seam for the post-result grace window (see STEER_GRACE_MS).
  steerGraceMs?: number
  signal?: AbortSignal
}

export interface ClaudeSystemPromptFile {
  path: string
  cleanup(): Promise<void>
}

const AUTH_FAILURE = /(not\s+logged\s+in|please\s+(?:run\s+)?(?:claude\s+)?(?:auth|login|\/login)|unauthori[sz]ed|token\s+(?:expired|invalid)|authentication\s+(?:failed|required)|failed\s+to\s+authenticate|invalid\s+authentication|\b401\b|session\s+expired|re-?auth|credentials?\s+(?:invalid|expired|missing))/i
const SILENCE_TIMEOUT_MS = 180_000
const TOTAL_TIMEOUT_MS = 1_200_000
const CLAUDE_SYSTEM_PROMPT_PREFIX = 'vontra-claude-prompt-'
const LEGACY_PROMPT_MAX_AGE_MS = 24 * 60 * 60 * 1000

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error: any) {
    // EPERM proves the process exists; only ESRCH proves it is gone.
    return error?.code !== 'ESRCH'
  }
}

// Remove plaintext prompt handoffs abandoned by a crashed TensorFold process.
export async function cleanupStaleClaudeSystemPrompts(
  root = tmpdir(),
  options: { currentPid?: number; now?: number; isProcessAlive?: (pid: number) => boolean } = {},
): Promise<void> {
  const currentPid = options.currentPid ?? process.pid
  const now = options.now ?? Date.now()
  const alive = options.isProcessAlive ?? processIsAlive
  const entries = await readdir(root, { withFileTypes: true }).catch(() => [])
  await Promise.all(entries.map(async (entry) => {
    if (!entry.isDirectory() || !entry.name.startsWith(CLAUDE_SYSTEM_PROMPT_PREFIX)) return
    const dir = path.join(root, entry.name)
    const match = /^vontra-claude-prompt-(\d+)-/.exec(entry.name)
    if (match) {
      const ownerPid = Number(match[1])

      if (ownerPid === currentPid || alive(ownerPid)) return
      await rm(dir, { recursive: true, force: true }).catch(() => { })
      return
    }

    const info = await stat(dir).catch(() => null)
    if (info && now - info.mtimeMs > LEGACY_PROMPT_MAX_AGE_MS) {
      await rm(dir, { recursive: true, force: true }).catch(() => { })
    }
  }))
}

export function claudeResultError(event: any): string | null {
  if (event && event.type === 'result' && event.is_error) {
    const msg = event.result ?? event.error ?? event.message
    return typeof msg === 'string' && msg.trim() ? msg : 'Claude CLI reported an error.'
  }
  return null
}

export function claudeFailureMessage(opts: { resultError?: string | null; exitCode: number; stderr: string }): string | null {
  const resultError = opts.resultError && opts.resultError.trim() ? opts.resultError.trim() : null
  if (!resultError && opts.exitCode === 0) return null
  const detail = (resultError || opts.stderr || '').trim()
  if (AUTH_FAILURE.test(detail)) {
    return 'Claude CLI is not authenticated. Run `claude login` (or `/login` inside Claude), then retry this provider.'
  }
  if (resultError) return `Claude CLI reported an error: ${detail.slice(0, 400)}`
  return `Claude CLI exited with code ${opts.exitCode}${detail ? `: ${detail.slice(0, 400)}` : ''}`
}

function binary(provider: Provider): string {
  // Resolve to an absolute path so a GUI-launched app (stripped PATH) can still find the CLI.
  return resolveCliBinary(provider.cliPath, ['claude'])
}

export function claudeEffort(value?: string): string {
  const v = (value || '').toLowerCase()
  if (v === 'minimal' || v === 'low') return 'low'
  if (v === 'high') return 'high'
  if (v === 'xhigh' || v === 'extra-high' || v === 'extra high' || v === 'ultrathink' || v === 'ultracode') return 'xhigh'
  if (v === 'max') return 'max'
  return 'medium'
}

export function applyClaudePromptEffortPrefix(text: string, effort?: string): string {
  const trimmed = text.trim()
  if (!trimmed) return trimmed
  if ((effort || '').toLowerCase() !== 'ultrathink') return trimmed
  if (trimmed.startsWith('Ultrathink:')) return trimmed
  return `Ultrathink:\n${trimmed}`
}

export function claudePermissionMode(provider: Provider, approvalMode?: ApprovalMode): string {
  if (approvalMode === 'readonly') return 'plan'
  // Auto may accept workspace edits, but must never select Claude's no-guard bypass mode.
  if (approvalMode === 'auto') return 'acceptEdits'
  if (provider.cliPermissionMode) return provider.cliPermissionMode
  return 'default'
}

export function claudeSessionIdFromEvent(event: any): string | null {
  const id = event?.session_id
  return typeof id === 'string' && id.trim() ? id.trim() : null
}

// One stream-json input line: the shape claude's `--input-format stream-json` accepts.
export function claudeUserMessageLine(text: string): string {
  return JSON.stringify({ type: 'user', message: { role: 'user', content: [{ type: 'text', text }] } }) + '\n'
}

const STEER_GRACE_MS = 1500

const RECOVERABLE_CLAUDE_RESUME = /(no conversation found|session.*(?:not found|does not exist|expired|invalid)|unknown session|could not resume|failed to resume)/i

export function isRecoverableClaudeResumeError(message: string): boolean {
  return RECOVERABLE_CLAUDE_RESUME.test(message)
}

export function shouldRetryFreshAfterResumeFailure(opts: { resumed: boolean; streamedContent: boolean; message: string }): boolean {
  return opts.resumed && !opts.streamedContent && isRecoverableClaudeResumeError(opts.message)
}

export function latestUserBody(messages: ApiMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return contentText(messages[i].content).trim()
  }
  return ''
}

export async function createClaudeSystemPromptFile(text: string, root = tmpdir(), pid = process.pid): Promise<ClaudeSystemPromptFile> {
  await cleanupStaleClaudeSystemPrompts(root, { currentPid: pid })
  const dir = await mkdtemp(path.join(root, `${CLAUDE_SYSTEM_PROMPT_PREFIX}${pid}-`))
  try {
    await chmod(dir, 0o700)
    const file = path.join(dir, 'system-prompt.txt')
    await writeFile(file, text, { encoding: 'utf8', flag: 'wx', mode: 0o600 })
    // Defend against an unexpectedly permissive process umask/platform implementation.
    await chmod(file, 0o600)
    let cleaned = false
    return {
      path: file,
      cleanup: async () => {
        if (cleaned) return
        cleaned = true
        await rm(dir, { recursive: true, force: true }).catch(() => { })
      },
    }
  } catch (error) {
    await rm(dir, { recursive: true, force: true }).catch(() => { })
    throw error
  }
}

function splitMessages(messages: ApiMessage[]): { system: string; body: string } {
  const systemParts: string[] = []
  const bodyParts: string[] = []
  for (const message of messages) {
    const role = message.role || 'user'
    const content = contentText(message.content).trim()
    if (!content && !message.tool_calls?.length) continue
    if (role === 'system') systemParts.push(content)
    else if (role === 'assistant') {
      const calls = message.tool_calls?.length
        ? `\n\n[assistant requested tools]\n${message.tool_calls.map((tc) => `${tc.function.name}: ${tc.function.arguments}`).join('\n')}`
        : ''
      bodyParts.push(`[assistant]\n${content}${calls}`)
    } else if (role === 'tool') {
      bodyParts.push(`[tool result: ${message.name || message.tool_call_id || 'tool'}]\n${content}`)
    } else {
      bodyParts.push(content)
    }
  }
  return { system: systemParts.join('\n\n').trim(), body: bodyParts.join('\n\n').trim() }
}

export function buildClaudeArgs(params: ClaudeCliParams, systemPromptFile: string | null, resume: ClaudeResumeCursor | null): string[] {
  const args = [
    '-p',

    '--input-format', 'stream-json',
    '--output-format', 'stream-json',
    '--include-partial-messages',
    '--verbose',
    '--permission-mode', claudePermissionMode(params.provider, params.approvalMode),
    '--effort', claudeEffort(params.reasoningEffort),
  ]
  if (params.model && params.model !== 'claude-code-default' && isSafeArgvId(params.model)) args.push('--model', params.model)
  if (params.provider.cliProfile?.trim()) args.push('--settings', params.provider.cliProfile.trim())
  if (resume?.sessionId && isSafeArgvId(resume.sessionId)) args.push('--resume', resume.sessionId)

  else if (systemPromptFile) args.push('--append-system-prompt-file', systemPromptFile)
  return args
}

export function textFromEvent(event: any, alreadyStreamed: boolean): string {
  const type = String(event?.type || '')
  if (type === 'stream_event') {
    const inner = event.event || {}
    if (inner.type === 'content_block_delta') {
      const delta = inner.delta || {}
      if (delta.type === 'text_delta' && typeof delta.text === 'string') return delta.text
    }
    return ''
  }
  if (type === 'content_block_delta') {
    const delta = event.delta || {}
    if (delta.type === 'text_delta' && typeof delta.text === 'string') return delta.text
  }
  if (type === 'assistant') {
    if (alreadyStreamed) return ''
    const blocks = event.message?.content || event.content || []
    if (Array.isArray(blocks)) return blocks.map((b) => typeof b === 'string' ? b : b?.text || '').join('')
  }
  if (!alreadyStreamed && type === 'result' && !event.is_error && typeof event.result === 'string') return event.result
  return ''
}

function compactStatusValue(value: string, max = 120): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

export function statusFromEvent(event: any): string {
  const type = String(event?.type || '')
  const inner = type === 'stream_event' ? event.event || {} : event
  if (inner.type === 'content_block_start') {
    const block = inner.content_block || {}
    if (block.type === 'tool_use' || block.type === 'server_tool_use') {
      return `Claude using ${compactStatusValue(String(block.name || 'tool'))}`
    }
    if (block.type === 'tool_result') return 'Claude received tool result'
    if (block.type === 'thinking') return 'Claude thinking'
  }
  if (inner.type === 'message_start') return 'Claude started a message'
  if (inner.type === 'message_stop') return 'Claude finished a message'
  if (type === 'system') return 'Claude session ready'
  return ''
}

export async function checkClaudeCli(provider: Provider): Promise<string> {
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
      reject(new Error('claude --version timed out'))
    }, 10_000)
    proc.stdout.on('data', (d) => { out += String(d) })
    proc.stderr.on('data', (d) => { err += String(d) })
    proc.on('error', (e) => {
      clearTimeout(timeout)
      reject(new Error(`Claude CLI not found at ${bin}: ${e.message}`))
    })
    proc.on('close', (code) => {
      clearTimeout(timeout)
      if (code === 0) resolve(out.trim() || 'Claude CLI ready')
      else reject(new Error((err || out || `claude --version exited ${code}`).trim()))
    })
  })
}

export async function* streamClaudeCli(params: ClaudeCliParams): AsyncGenerator<ClaudeCliChunk> {
  const resume = params.resumeCursor?.sessionId ? params.resumeCursor : null
  if (resume) {
    let streamedContent = false
    try {
      for await (const chunk of runClaudeOnce(params, resume)) {
        if (chunk.type === 'content') streamedContent = true
        yield chunk
      }
      return
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (!shouldRetryFreshAfterResumeFailure({ resumed: true, streamedContent, message })) throw error
    }
  }
  yield* runClaudeOnce(params, null)
}

async function* runClaudeOnce(params: ClaudeCliParams, resume: ClaudeResumeCursor | null): AsyncGenerator<ClaudeCliChunk> {
  const { system, body } = splitMessages(params.messages)

  const effectiveBody = resume ? (latestUserBody(params.messages) || body) : body
  if (!effectiveBody || params.signal?.aborted) return
  const promptBody = applyClaudePromptEffortPrefix(effectiveBody, params.reasoningEffort)

  const systemPromptFile = !resume && system ? await createClaudeSystemPromptFile(system, cliConfinement(params.provider)?.tmpDir) : null
  try {

    const bin = binary(params.provider)
    const args = buildClaudeArgs(params, systemPromptFile?.path ?? null, resume)
    const run = planningCommand(params.provider, bin, args, { cwd: params.cwd || undefined, env: cliSpawnEnv(params.provider.cliEnv) })
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
      if (now - startedAt > TOTAL_TIMEOUT_MS) watchdogKilled = 'Claude CLI total timeout exceeded.'
      else if (now - lastOutputAt > SILENCE_TIMEOUT_MS) watchdogKilled = `Claude CLI was silent for ${Math.round(SILENCE_TIMEOUT_MS / 1000)}s.`
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

    let stdinClosed = false
    const closeStdin = () => {
      if (stdinClosed) return
      stdinClosed = true
      try { proc.stdin.end() } catch { }
    }
    const writeUserLine = (text: string): boolean => {
      if (stdinClosed || proc.exitCode != null) return false
      try {
        proc.stdin.write(claudeUserMessageLine(text))
        return true
      } catch {
        return false
      }
    }
    let steersSinceResult = 0
    let graceTimer: ReturnType<typeof setTimeout> | null = null
    const clearGrace = () => {
      if (graceTimer) {
        clearTimeout(graceTimer)
        graceTimer = null
      }
    }
    const disconnectSteering = params.steering?.connect(
      (input) => {
        const wrote = writeUserLine(input.text)
        if (wrote) steersSinceResult++
        return wrote
      },
      () => !stdinClosed && proc.exitCode == null,
    )
    writeUserLine(promptBody)
    if (!params.steering) closeStdin()

    let resultError: string | null = null
    let streamed = false
    let completed = false
    let sessionId: string | null = resume?.sessionId ?? null
    const lines = readline.createInterface({ input: proc.stdout })
    try {
      for await (const line of lines) {
        if (params.signal?.aborted) break
        const raw = String(line || '').trim()
        if (!raw) continue
        lastOutputAt = Date.now()
        let event: any
        try { event = JSON.parse(raw) } catch { continue }
        resultError = claudeResultError(event) ?? resultError
        sessionId = claudeSessionIdFromEvent(event) ?? sessionId
        if (params.steering && !stdinClosed) {
          if (event?.type === 'result') {
            const hadPendingSteers = steersSinceResult > 0
            steersSinceResult = 0
            // No steer in flight → the turn is over and nothing else can start one.
            if (!hadPendingSteers) closeStdin()

            else if (!graceTimer) graceTimer = setTimeout(closeStdin, params.steerGraceMs ?? STEER_GRACE_MS)
          } else if (graceTimer) {
            clearGrace()
          }
        }
        const status = statusFromEvent(event)
        if (status) yield { type: 'reasoning', text: `${status}\n` }
        const text = textFromEvent(event, streamed)
        if (text) {
          streamed = true
          yield { type: 'content', text }
        }
      }
      completed = true // stdout reached EOF naturally; the exit-code await below is valid
    } finally {
      clearGrace()
      disconnectSteering?.()
      closeStdin()
      clearInterval(watchdog)
      params.signal?.removeEventListener('abort', abort)

      if (!completed) lines.close()
      if (!completed && proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
    }

    const exitCode: number = await new Promise((resolve) => {
      if (proc.exitCode != null) resolve(proc.exitCode)
      else proc.once('close', (code) => resolve(code ?? 0))
    })
    if (params.signal?.aborted) return
    if (spawnErrorMessage) throw new Error(`Claude CLI not found at ${bin}: ${spawnErrorMessage}`)
    if (watchdogKilled) throw new Error(watchdogKilled)
    const failure = claudeFailureMessage({ resultError, exitCode, stderr })
    if (failure) throw new Error(failure)
    if (!streamed) throw new Error('Claude CLI finished without a final message.')
    if (sessionId) yield { type: 'resume_cursor', cursor: { sessionId } }
  } finally {
    await systemPromptFile?.cleanup()
  }
}
