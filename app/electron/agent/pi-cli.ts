

import { randomUUID } from 'node:crypto'
import type { ChildProcessWithoutNullStreams } from 'node:child_process'
import { spawnCliProcess as spawn } from './cli-process'
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

// Remove private prompt directories abandoned by a crashed TensorFold process.
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

export function piThinkingLevel(value?: string): string {
  const v = (value || '').toLowerCase()
  if (v === 'off' || v === 'none') return 'off'
  if (v === 'minimal') return 'minimal'
  if (v === 'low') return 'low'
  if (v === 'high') return 'high'
  if (v === 'xhigh' || v === 'extra-high' || v === 'extra high' || v === 'max' || v === 'ultrathink' || v === 'ultracode') return 'xhigh'
  return 'medium'
}

export function piToolsForApproval(approvalMode?: ApprovalMode): string[] | null {
  if (approvalMode === 'readonly') return ['read']
  return null
}

export function buildPiArgs(params: PiCliParams, systemPromptFile: string, sessionId: string, resuming: boolean): string[] {
  assertPiCredentialHandoff(params.provider)
  const args = ['--mode', 'rpc', '--thinking', piThinkingLevel(params.reasoningEffort)]
  if (params.model && params.model !== 'pi-default' && isSafeArgvId(params.model)) args.push('--model', params.model)
  const tools = piToolsForApproval(params.approvalMode)
  if (tools) args.push('--tools', tools.join(','))

  if (isSafeArgvId(sessionId)) args.push('--session-id', sessionId)
  // A resumed session already carries the system prompt from the run that created it.
  if (!resuming && systemPromptFile) args.push('--append-system-prompt', systemPromptFile)
  return args
}

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

export function piFailureMessage(opts: { exitCode: number; stderr: string; retryFinalError?: string | null; promptRejected?: string | null }): string | null {
  if (opts.promptRejected?.trim()) return `Pi rejected the prompt: ${opts.promptRejected.trim().slice(0, 400)}`
  if (opts.retryFinalError?.trim()) return `Pi gave up after retries: ${opts.retryFinalError.trim().slice(0, 400)}`
  if (opts.exitCode === 0) return null
  const detail = opts.stderr.trim()
  return `Pi CLI exited with code ${opts.exitCode}${detail ? `: ${detail.slice(0, 400)}` : ''}`
}

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
    try { proc.stdin.end() } catch {  }
  }

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
      try { admitEvent(JSON.parse(raw)) } catch {  }
    }
    wakeQueue?.()
  })
  proc.stdout.on('end', () => {
    if (lineBuffer.trim()) {
      try { admitEvent(JSON.parse(lineBuffer)) } catch {  }
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
