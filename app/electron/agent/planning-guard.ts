import os from 'node:os'
import path from 'node:path'
import { classifyProvider } from '../../shared/provider-trust'
import { PlaceholderVault, RedactionRun, createStreamRehydrator, describeWithheld, withheldTotal, type RedactionContext } from './outbound-redaction'
import { confineProvider, planningSandboxAvailable, preparePlanningDir, removePlanningDir, type CliConfinement } from './planning-sandbox'
import type { EngineSteeringChannel } from './steering'
import { contentText } from './content.ts'
import type { ApiMessage, ApprovalMode, Provider } from './types'

// Planning-only mode for external providers (cloud HTTP endpoints and every delegated CLI).
// One entry point wraps a call: it strips tool schemas, forces read-only approval, swaps
// the workspace for an empty planning folder (CLIs also run inside a seatbelt, see
// planning-sandbox.ts), redacts what is sent, and rehydrates placeholders in the reply.

export interface GuardableParams {
  provider?: Provider
  baseUrl?: string
  messages: ApiMessage[]
  tools?: unknown[]
  approvalMode?: ApprovalMode
  cwd?: string | null
  threadId?: string
  steering?: EngineSteeringChannel
}

export type PlanningNoticeChunk = { type: 'notice'; text: string }
type TextChunk = { type: string; text?: string }

export const PLANNING_NOTE = [
  'Planning-only mode. You are an external model helping the user think and plan.',
  "You have no tools and no access to the user's files, shell or machine, so do not ask to run commands or open files.",
  "Answer or advise in text. When the user wants work done, write a concrete step-by-step plan; the user's local TensorFold engine will carry it out.",
  'Placeholders such as [EMAIL_1], [TERM_2] or [WORKSPACE_1] stand for values withheld from you. Keep them exactly as written.',
].join('\n')

export const PROJECT_CONTEXT_WITHHELD_NOTE = 'Project guidance files and project memory were withheld from this external model.'

// Backstop for prompts built elsewhere: loop.ts already leaves these blocks out for planners.
const PROJECT_CONTEXT_BLOCKS = [
  /\n*PROJECT GUIDANCE DATA \(from [^\n]*\n[\s\S]*?(?=\n\nDURABLE PROJECT MEMORY|$)/g,
  /\n*DURABLE PROJECT MEMORY[\s\S]*?(?:<\/project_memory>|$)/g,
]

// Delegated agent CLIs that run against a whole folder with their own cloud model.
const AGENT_CLI_IDS = new Set(['claude', 'codex', 'cursor', 'gemini', 'grok', 'hermes', 'opencode', 'kilo', 'pi', 'omp'])
const AGENT_CLI_BINARIES = new Set(['claude', 'codex', 'cursor-agent', 'gemini', 'grok', 'hermes', 'opencode', 'kilo', 'pi', 'omp', 'aider', 'amp', 'goose', 'qwen', 'crush'])

export const OMP_ENGINE_REFUSAL = "The OMP engine picks its own models, cloud ones included, and reads this workspace itself, so TensorFold can't keep confidential work local. Switch this thread to the built-in engine."

/**
 * OMP takes its models from its own config (any provider, role models such as smol/slow/plan,
 * switchable mid-run) and reads the workspace natively. Nothing here can prove those models are
 * trusted-local, so OMP thread runs are refused.
 */
export function assertOmpEngineAllowed(): void {
  throw new Error(OMP_ENGINE_REFUSAL)
}

const MAX_VAULTS = 64
const guarded = new WeakSet<object>()
const vaults = new Map<string, PlaceholderVault>()
let confidentialTerms: () => readonly string[] = () => []

export function configurePlanningGuard(options: { confidentialTerms: () => readonly string[] | null | undefined }): void {
  confidentialTerms = () => options.confidentialTerms() ?? []
}

export function needsPlanningGuard(p: { provider?: Provider; baseUrl?: string }): boolean {
  return !guarded.has(p) && classifyProvider(p.provider, p.baseUrl) === 'external'
}

function vaultFor(threadId?: string): PlaceholderVault {
  if (!threadId) return new PlaceholderVault()
  const vault = vaults.get(threadId) ?? new PlaceholderVault()
  vaults.delete(threadId)
  vaults.set(threadId, vault)
  if (vaults.size > MAX_VAULTS) vaults.delete(vaults.keys().next().value as string)
  return vault
}

export function redactionContext(workspace?: string | null, terms: readonly string[] = confidentialTerms()): RedactionContext {
  let username = ''
  try { username = os.userInfo().username } catch { /* no passwd entry */ }
  return { workspace, homeDir: os.homedir(), username, hostname: os.hostname(), terms }
}

function withoutProjectContext(messages: readonly ApiMessage[], redaction: RedactionRun): ApiMessage[] {
  return messages.map((message) => {
    if (message.role !== 'system' || typeof message.content !== 'string') return message
    let removed = 0
    const content = PROJECT_CONTEXT_BLOCKS.reduce((text, re) => text.replace(re, () => { removed++; return '' }), message.content)
    if (!removed) return message
    redaction.tally.project_context = (redaction.tally.project_context ?? 0) + removed
    return { ...message, content: `${content}\n\n${PROJECT_CONTEXT_WITHHELD_NOTE}` }
  })
}

/** Why a project terminal must not launch this CLI, or null for a shell or another tool. */
export function agentTerminalRefusal(cli: { id: string; name?: string; command?: string | null }): string | null {
  const binary = path.basename(String(cli.command ?? '').trim().split(/\s+/)[0] ?? '')
  if (!AGENT_CLI_IDS.has(cli.id) && !AGENT_CLI_BINARIES.has(binary)) return null
  return `${cli.name || binary} sends this folder to an external model, so TensorFold doesn't launch it in a project terminal. Pick it in the composer, where external models only plan, or open a plain shell.`
}

// The note joins the first system message, which leads: strict chat templates refuse a second system message (#884).
function withPlanningNote(messages: ApiMessage[]): ApiMessage[] {
  const first = messages.findIndex((message) => message.role === 'system')
  if (first < 0) return [{ role: 'system', content: PLANNING_NOTE }, ...messages]
  const system = messages[first]
  const content: ApiMessage['content'] = Array.isArray(system.content)
    ? [...system.content, { type: 'text', text: PLANNING_NOTE }]
    : system.content ? `${system.content}\n\n${PLANNING_NOTE}` : PLANNING_NOTE
  return [{ ...system, content }, ...messages.slice(0, first), ...messages.slice(first + 1)]
}

function redactedSteering(channel: EngineSteeringChannel, redact: (text: string) => string): EngineSteeringChannel {
  return {
    connect: (consumer, isReady) => channel.connect((input) => consumer({ ...input, text: redact(input.text) }), isReady),
    get canAccept() { return channel.canAccept },
    offer: (input) => channel.offer(input),
  }
}

/** The latest assistant reply in a thread: the plan a 'run-plan' handoff carries out. */
export function latestPlanText(messages: readonly ApiMessage[] | null | undefined): string {
  for (const message of [...(messages ?? [])].reverse()) {
    const text = message.role === 'assistant' ? contentText(message.content).trim() : ''
    if (text) return text
  }
  return ''
}

/** First turn for the local engine that runs an external model's (rehydrated) plan. */
export function runPlanPrompt(fromEngine: string, plan: string): string {
  return `Carry out this plan from ${fromEngine} in the workspace, step by step with your tools. Stop and ask if a step is unclear or unsafe.\n\n<plan>\n${plan.slice(0, 20_000)}\n</plan>`
}

/** Preview for Settings: what an external provider would receive for `text`. */
export function previewWithheld(text: string, terms?: readonly string[]): { text: string; withheld: string; total: number } {
  const run = new RedactionRun(new PlaceholderVault(), redactionContext(null, terms ?? confidentialTerms()))
  const out = run.text(String(text ?? '').slice(0, 20_000))
  return { text: out, withheld: describeWithheld(run.tally), total: withheldTotal(run.tally) }
}

/**
 * Guard one call. Returns null for trusted local providers (and for calls already guarded),
 * otherwise a stream to yield in place of `run(p)`.
 */
export function planningGuard<P extends GuardableParams, C extends TextChunk>(p: P, run: (p: P) => AsyncIterable<C>): AsyncGenerator<C | PlanningNoticeChunk> | null {
  if (!needsPlanningGuard(p)) return null
  return guardedStream(p, run)
}

async function* guardedStream<P extends GuardableParams, C extends TextChunk>(p: P, run: (p: P) => AsyncIterable<C>): AsyncGenerator<C | PlanningNoticeChunk> {
  const transport = p.provider?.transport ?? 'chat-completions'
  const label = p.provider?.name || p.provider?.id || 'the external model'
  if (transport === 'opencode' || transport === 'kilo') {
    throw new Error(`${label} runs as a separate server that TensorFold cannot confine, so it can't be used for planning-only work. Choose another provider.`)
  }
  const isCli = transport !== 'chat-completions'
  if (isCli && !planningSandboxAvailable()) {
    throw new Error(`Planning-only mode for ${label} needs the macOS sandbox (sandbox-exec), which is not available here, so nothing was sent.`)
  }
  const vault = vaultFor(p.threadId)
  const context = redactionContext(p.cwd)
  const redaction = new RedactionRun(vault, context)
  const messages = withPlanningNote(redaction.messages(withoutProjectContext(p.messages, redaction)))
  const confinement: CliConfinement | null = isCli
    ? { ...preparePlanningDir(p.threadId ?? null), denyRoots: p.cwd ? [p.cwd] : [] }
    : null
  const next = {
    ...p,
    messages,
    tools: undefined,
    approvalMode: 'readonly' as const,
    cwd: confinement?.planningDir ?? null,
    provider: p.provider && confinement ? confineProvider(p.provider, confinement) : p.provider,
    steering: p.steering ? redactedSteering(p.steering, (text) => new RedactionRun(vault, context).text(text)) : undefined,
  } as P
  guarded.add(next)

  const total = withheldTotal(redaction.tally)
  if (total) yield { type: 'notice', text: `Sent to ${label} with ${total} item${total === 1 ? '' : 's'} withheld (${describeWithheld(redaction.tally)}).` }

  const rehydrators: Record<string, ReturnType<typeof createStreamRehydrator>> = {
    content: createStreamRehydrator(vault),
    reasoning: createStreamRehydrator(vault),
  }
  function* flush(): Generator<C> {
    for (const type of ['reasoning', 'content']) {
      const rest = rehydrators[type].flush()
      if (rest) yield { type, text: rest } as C
    }
  }
  try {
    for await (const chunk of run(next)) {
      const rehydrator = rehydrators[chunk.type]
      if (rehydrator) {
        const text = rehydrator.push(chunk.text ?? '')
        if (text) yield { ...chunk, text }
        continue
      }
      // A planner never acts: drop any tool call a provider sends without schemas.
      if (chunk.type === 'tool_calls') continue
      yield* flush()
      yield chunk
    }
    yield* flush()
  } finally {
    if (confinement && !p.threadId) removePlanningDir(confinement)
  }
}
