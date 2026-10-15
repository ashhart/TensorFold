// Gemini CLI engine (VST-033): drive `gemini` headless as a thread engine. Wire facts from
// google-gemini/gemini-cli's shipped docs (docs/cli/{headless,cli-reference}.md) — the
// binary is not installed on this machine, so only DOCUMENTED behavior is used:
// - Piped stdin triggers headless mode; `--output-format json` emits ONE JSON object:
//     { "response": "<text>", "stats": {…}, "error": {…}? }
//   (stream-json exists but its event FIELD shapes aren't documented with examples, so v1
//   deliberately uses the fully-specified json mode; the delegated-CLI heartbeat keeps the
//   UI alive during the silent wait.)
// - `-m/--model` picks the model; `--approval-mode default|auto_edit|yolo|plan` gates tools.
// - Gemini's CLI reference says `--prompt` is appended to stdin input, so the complete
//   conversation is sent only through stdin and never copied into process argv.
// - `--resume/-r` exists but its ARGUMENT semantics are undocumented — no resume cursor in
//   v1; every turn sends the flattened conversation (like claude-cli before its resume
//   upgrade). The documented API-key handoff is GEMINI_API_KEY.
// - Exit codes: 0 ok, 1 general, 42 input error, 53 turn limit.

import { spawn } from 'node:child_process'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { resolveCliBinary } from './cli-path'
import { cliSpawnEnv } from './cli-spawn-env'
import { planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { ApiMessage, ApprovalMode, Provider } from './types'

export type GeminiCliChunk =
  | { type: 'content'; text: string }
  | { type: 'reasoning'; text: string }

export interface GeminiCliParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  cwd?: string | null
  approvalMode?: ApprovalMode
  signal?: AbortSignal
}

const TOTAL_TIMEOUT_MS = 1_200_000

function binary(provider: Provider): string {
  return resolveCliBinary(provider.cliPath, ['gemini'])
}

// TensorFold approval modes → gemini's documented guarded modes. Full Auto permits automatic
// workspace edits but never selects yolo, which bypasses Gemini's safety prompts entirely.
export function geminiApprovalMode(approvalMode?: ApprovalMode): 'default' | 'auto_edit' | 'yolo' | 'plan' {
  if (approvalMode === 'readonly') return 'plan'
  if (approvalMode === 'auto') return 'auto_edit'
  return 'default'
}

export function buildGeminiArgs(params: GeminiCliParams): string[] {
  const args = ['--output-format', 'json', '--approval-mode', geminiApprovalMode(params.approvalMode)]
  if (params.model && params.model !== 'gemini-default') args.push('-m', params.model)
  return args
}

/** Map TensorFold's raw Gemini key to the CLI's documented environment variable, never argv. */
export function geminiSpawnEnv(provider: Provider, inherited: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const apiKey = provider.apiKey?.trim()
  // Allowlisted basics + provider env only; the raw key rides the documented GEMINI_API_KEY.
  return cliSpawnEnv({ ...(provider.cliEnv ?? {}), ...(apiKey ? { GEMINI_API_KEY: apiKey } : {}) }, inherited)
}

// Parse the single documented output object. Defensive on the error shape (its inner field
// names aren't documented): prefer .message, fall back to a bounded JSON slice.
export function parseGeminiOutput(stdout: string): { response: string; errorMessage: string | null } {
  const text = stdout.trim()
  if (!text) return { response: '', errorMessage: null }
  // The object may be preceded by stray non-JSON warnings; take the first '{' onward.
  const start = text.indexOf('{')
  if (start < 0) return { response: '', errorMessage: null }
  try {
    const parsed = JSON.parse(text.slice(start))
    const response = typeof parsed?.response === 'string' ? parsed.response : ''
    let errorMessage: string | null = null
    if (parsed?.error) {
      const err = parsed.error
      errorMessage = typeof err === 'string' ? err
        : typeof err?.message === 'string' ? err.message
        : JSON.stringify(err).slice(0, 300)
    }
    return { response, errorMessage }
  } catch {
    return { response: '', errorMessage: null }
  }
}

export function geminiFailureMessage(opts: { exitCode: number; stderr: string; errorMessage: string | null }): string | null {
  if (opts.errorMessage) return `Gemini CLI reported an error: ${opts.errorMessage.slice(0, 400)}`
  if (opts.exitCode === 0) return null
  const kind = opts.exitCode === 42 ? ' (input error)' : opts.exitCode === 53 ? ' (turn limit exceeded)' : ''
  const detail = opts.stderr.trim()
  return `Gemini CLI exited with code ${opts.exitCode}${kind}${detail ? `: ${detail.slice(0, 400)}` : ''}`
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

export async function* streamGeminiCli(params: GeminiCliParams): AsyncGenerator<GeminiCliChunk> {
  const prompt = flattenMessages(params.messages)
  if (!prompt || params.signal?.aborted) return

  const bin = binary(params.provider)
  const run = planningCommand(params.provider, bin, buildGeminiArgs(params), { cwd: params.cwd || undefined, env: geminiSpawnEnv(params.provider) })
  const proc = spawn(run.command, run.args, {
    cwd: run.cwd,
    detached: detachedProcessGroup(),
    env: run.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  let spawnErrorMessage = ''
  let watchdogKilled = ''
  const startedAt = Date.now()
  const watchdog = setInterval(() => {
    if (proc.exitCode != null) return
    if (Date.now() - startedAt > TOTAL_TIMEOUT_MS) {
      watchdogKilled = 'Gemini CLI total timeout exceeded.'
      terminateChildProcess(proc, { processGroup: true })
    }
  }, 5_000)
  proc.once('error', (e) => { spawnErrorMessage = e.message })
  proc.stdin.on('error', () => { /* child exit/spawn failure is reported below */ })
  try { proc.stdin.end(prompt) } catch { /* spawn failure is reported below */ }

  const abort = () => {
    if (proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }
  params.signal?.addEventListener('abort', abort, { once: true })

  let stdout = ''
  let stderr = ''
  proc.stdout.on('data', (d) => { stdout += String(d) })
  proc.stderr.on('data', (d) => { stderr = (stderr + String(d)).slice(-4000) })

  let completed = false
  let exitCode = -1
  try {
    yield { type: 'reasoning', text: 'Gemini working…\n' }
    exitCode = await new Promise((resolve) => {
      proc.once('close', (code) => resolve(code ?? 0))
      proc.once('error', () => resolve(-1))
    })
    completed = true
  } finally {
    clearInterval(watchdog)
    params.signal?.removeEventListener('abort', abort)
    // The initial activity chunk is yielded while Gemini is still alive. If
    // the consumer stops there, reap the group instead of orphaning the run.
    if (!completed && proc.exitCode == null) terminateChildProcess(proc, { processGroup: true })
  }
  if (params.signal?.aborted) return
  if (spawnErrorMessage) throw new Error(`Gemini CLI not found at ${bin}: ${spawnErrorMessage}`)
  if (watchdogKilled) throw new Error(watchdogKilled)

  const { response, errorMessage } = parseGeminiOutput(stdout)
  const failure = geminiFailureMessage({ exitCode, stderr, errorMessage })
  if (failure) throw new Error(failure)
  if (!response.trim()) throw new Error('Gemini CLI finished without a response.')
  yield { type: 'content', text: response }
}
