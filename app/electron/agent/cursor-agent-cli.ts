// Cursor Agent engine (VST-033): drive `cursor-agent` headless as a thread engine. Wire
// facts from Cursor's CLI reference (docs/cli/reference/{parameters,output-format}) — the
// binary is not installed on this machine, so shapes come from the published docs:
// - `cursor-agent -p --output-format stream-json` emits NDJSON events:
//     {type:"system",subtype:"init",session_id,model,...}      (once, session id → cursor)
//     {type:"assistant",message:{role,content:[{type:"text",text}]},session_id}
//     {type:"tool_call",subtype:"started"|"completed",call_id,tool_call:{<name>ToolCall:…}}
//     {type:"result",subtype,is_error,result,session_id}       (terminal)
// - `--resume <chatId>` resumes a session; `--model <model>`; CURSOR_API_KEY authenticates
//   without exposing the credential in the child process argument list.
// - `--stream-partial-output` makes assistant events stream partial text. The docs don't
//   pin down whether partials are cumulative or incremental, so `cursorTextFromAssistant`
//   handles BOTH: text extending what we've emitted yields the suffix; otherwise the new
//   text is emitted whole. Without the flag the single complete assistant event and the
//   terminal result both carry full text — the same dedup makes the result a no-op then.

import { spawn } from 'node:child_process'
import readline from 'node:readline'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { resolveCliBinary, isSafeArgvId } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { ApiMessage, ApprovalMode, Provider } from './types'

export interface CursorResumeCursor {
  sessionId: string
}

export type CursorCliChunk =
  | { type: 'content'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'resume_cursor'; cursor: CursorResumeCursor }

export interface CursorCliParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  cwd?: string | null
  approvalMode?: ApprovalMode
  resumeCursor?: CursorResumeCursor | null
  signal?: AbortSignal
}

const SILENCE_TIMEOUT_MS = 180_000
const TOTAL_TIMEOUT_MS = 1_200_000

function binary(provider: Provider): string {
  return resolveCliBinary(provider.cliPath, ['cursor-agent'])
}

export function buildCursorArgs(params: CursorCliParams, resume: CursorResumeCursor | null): string[] {
  const args = ['-p', '--output-format', 'stream-json', '--stream-partial-output']
  // Cursor print mode can edit files and run shell commands without a TensorFold approval
  // bridge. Only explicit Full Auto may use that writable posture; every other mode
  // uses Cursor's documented read-only ask mode.
  if (params.approvalMode !== 'auto') args.push('--mode=ask')
  if (params.model && params.model !== 'cursor-default' && isSafeArgvId(params.model)) args.push('--model', params.model)
  if (resume?.sessionId && isSafeArgvId(resume.sessionId)) args.push('--resume', resume.sessionId)
  return args
}

export function cursorSpawnEnv(provider: Provider, inherited: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const apiKey = provider.apiKey?.trim()
  // Allowlisted basics + provider env only; the raw key rides the documented CURSOR_API_KEY.
  return cliSpawnEnv({ ...(provider.cliEnv ?? {}), ...(apiKey ? { CURSOR_API_KEY: apiKey } : {}) }, inherited)
}

// Session id can arrive on the init event or any later event; first non-empty wins.
export function cursorSessionIdFromEvent(event: any): string | null {
  const id = event?.session_id
  return typeof id === 'string' && id.trim() ? id.trim() : null
}

// Extract the assistant text of an event's content blocks.
function assistantText(event: any): string {
  const blocks = event?.message?.content
  if (!Array.isArray(blocks)) return ''
  return blocks.map((block: any) => (block && block.type === 'text' && typeof block.text === 'string') ? block.text : '').join('')
}

// Cumulative-or-incremental dedup: given everything emitted so far, return only the NEW
// text this assistant event contributes. Cumulative partials extend the accumulator (emit
// the suffix); incremental partials don't (emit whole); an exact repeat (the final assistant
// event / terminal result re-carrying full text) emits nothing.
export function cursorTextFromAssistant(event: any, emittedSoFar: string): string {
  const text = assistantText(event)
  if (!text) return ''
  if (text === emittedSoFar) return ''
  if (text.startsWith(emittedSoFar)) return text.slice(emittedSoFar.length)
  if (emittedSoFar.endsWith(text) || emittedSoFar.includes(text)) return ''
  return text
}

function compactToolName(event: any): string {
  const call = event?.tool_call
  if (call && typeof call === 'object') {
    const key = Object.keys(call)[0] ?? ''
    return key.replace(/ToolCall$/, '') || 'tool'
  }
  return 'tool'
}

// Status line for tool lifecycle — tool NAME only, never args/results.
export function cursorStatusFromEvent(event: any): string {
  if (event?.type === 'tool_call' && event.subtype === 'started') return `Cursor using ${compactToolName(event)}`
  if (event?.type === 'system' && event.subtype === 'init') return 'Cursor session ready'
  return ''
}

// Terminal result classification: {type:'result', is_error, result}.
export function cursorResultError(event: any): string | null {
  if (event?.type === 'result' && event.is_error) {
    const msg = event.result ?? event.error ?? event.subtype
    return typeof msg === 'string' && msg.trim() ? msg : 'Cursor Agent reported an error.'
  }
  return null
}

export function cursorFailureMessage(opts: { resultError: string | null; exitCode: number; stderr: string }): string | null {
  if (opts.resultError) return `Cursor Agent reported an error: ${opts.resultError.slice(0, 400)}`
  if (opts.exitCode === 0) return null
  const detail = opts.stderr.trim()
  return `Cursor Agent exited with code ${opts.exitCode}${detail ? `: ${detail.slice(0, 400)}` : ''}`
}

export function latestCursorUserBody(messages: ApiMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return contentText(messages[i].content).trim()
  }
  return ''
}

function flattenMessages(messages: ApiMessage[]): string {
  const parts: string[] = []
  for (const message of messages) {
    const role = message.role || 'user'
    const content = contentText(message.content).trim()
    if (!content) continue
    if (role === 'system') parts.push(`[system]\n${content}`)
    else if (role === 'assistant') parts.push(`[assistant]\n${content}`)
    else if (role === 'tool') parts.push(`[tool result]\n${content}`)
    else parts.push(content)
  }
  return parts.join('\n\n').trim()
}

export async function* streamCursorAgent(params: CursorCliParams): AsyncGenerator<CursorCliChunk> {
  const resume = params.resumeCursor?.sessionId ? params.resumeCursor : null
  const body = resume ? (latestCursorUserBody(params.messages) || flattenMessages(params.messages)) : flattenMessages(params.messages)
  if (!body || params.signal?.aborted) return

  const bin = binary(params.provider)
  const run = planningCommand(params.provider, bin, buildCursorArgs(params, resume), { cwd: params.cwd || undefined, env: cursorSpawnEnv(params.provider) })
  const proc = spawn(run.command, run.args, {
    cwd: run.cwd,
    detached: detachedProcessGroup(),
    env: run.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  // Cursor documents piped stdin as a non-interactive prompt source. Keep the
  // full conversation out of argv, where same-user process inspection and crash
  // reporters could otherwise capture it.
  proc.stdin.on('error', () => { /* spawn/early-exit is reported through stderr/close */ })
  proc.stdin.end(body)

  let spawnErrorMessage = ''
  let watchdogKilled = ''
  let lastOutputAt = Date.now()
  const startedAt = lastOutputAt
  const watchdog = setInterval(() => {
    if (proc.exitCode != null) return
    const now = Date.now()
    if (now - startedAt > TOTAL_TIMEOUT_MS) watchdogKilled = 'Cursor Agent total timeout exceeded.'
    else if (now - lastOutputAt > SILENCE_TIMEOUT_MS) watchdogKilled = `Cursor Agent was silent for ${Math.round(SILENCE_TIMEOUT_MS / 1000)}s.`
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

  let sessionId: string | null = resume?.sessionId ?? null
  let resultError: string | null = null
  let emitted = ''
  let completed = false
  const lines = readline.createInterface({ input: proc.stdout })
  try {
    for await (const line of lines) {
      if (params.signal?.aborted) break
      const raw = String(line || '').trim()
      if (!raw) continue
      lastOutputAt = Date.now()
      let event: any
      try { event = JSON.parse(raw) } catch { continue }
      sessionId = cursorSessionIdFromEvent(event) ?? sessionId
      resultError = cursorResultError(event) ?? resultError
      const status = cursorStatusFromEvent(event)
      if (status) yield { type: 'reasoning', text: `${status}\n` }
      if (event?.type === 'assistant') {
        const text = cursorTextFromAssistant(event, emitted)
        if (text) {
          emitted += text
          yield { type: 'content', text }
        }
      } else if (event?.type === 'result' && !event.is_error && typeof event.result === 'string' && !emitted) {
        // No assistant events streamed (plain json-esque run): the terminal result carries
        // the full text.
        emitted = event.result
        yield { type: 'content', text: event.result }
      }
    }
    completed = true
  } finally {
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
  if (spawnErrorMessage) throw new Error(`Cursor Agent not found at ${bin}: ${spawnErrorMessage}`)
  if (watchdogKilled) throw new Error(watchdogKilled)
  const failure = cursorFailureMessage({ resultError, exitCode, stderr })
  if (failure) throw new Error(failure)
  if (!emitted) throw new Error('Cursor Agent finished without a final message.')
  if (sessionId) yield { type: 'resume_cursor', cursor: { sessionId } }
}
