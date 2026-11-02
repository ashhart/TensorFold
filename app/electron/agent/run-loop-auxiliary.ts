// Read-only goal checks and research children retain the parent workspace and live request guard.

import { streamChat } from './openai'
import { assertThreadChildSupported } from '../../shared/thread-child-policy'
import { executeTool, TOOL_SCHEMAS, buildToolSchemas, isReadOnlyTool, type ToolContext, type ToolResult } from './tools'
import { coerceToolArgs } from './tool-arg-coercion'
import { type DelegateTaskInput } from './delegate-task'
import { buildOracleUserMessage, buildOracleSystemPrompt, type OracleQuery } from './oracle'
import { renderTranscript } from './compaction'
import { compressToolOutput, rememberPreparedToolOutput } from './tool-output-compression'
import { ensureTensorFoldStudioStylePrompt } from './prompt-style'
import { type ApiMessage, type Provider, type Thread, type ToolCall } from './types'
import { wrapUntrustedToolOutput } from './run-loop-history'

export const MAX_GOAL_ITERS = 6

export async function verifyGoal(provider: Provider, model: string, thread: Thread, signal: AbortSignal, assertRequestAllowed?: () => void): Promise<{ done: boolean; note: string }> {
  assertRequestAllowed?.()
  const transcript = renderTranscript(thread.apiMessages.slice(-14))
  const sys = ensureTensorFoldStudioStylePrompt('You are a strict verifier for an AI coding agent. Given a GOAL and the recent work, decide if the goal is FULLY achieved and verified (not just attempted). Reply on ONE line, starting with either "DONE: <one-sentence reason>" or "CONTINUE: <what is still missing>".')
  let out = ''
  try {
    for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages: [{ role: 'system', content: sys }, { role: 'user', content: `GOAL\n${thread.goal}\n\nRECENT WORK\n${transcript}` }], temperature: 0.1, approvalMode: 'readonly', cwd: thread.cwd, signal, assertRequestAllowed })) {
      if (ch.type === 'content') out += ch.text
    }
  } catch { assertRequestAllowed?.(); return { done: false, note: 'verification failed; continuing' } }
  const t = out.trim()
  const done = /^\s*done\b/i.test(t)
  const note = t.replace(/^\s*(done|continue)\s*[:\-]?\s*/i, '').trim().slice(0, 200) || (done ? 'verified' : 'keep going')
  return { done, note }
}

const EXPLORE_TOOL_NAMES = ['list_dir', 'read_file', 'search', 'glob']

const EXPLORE_TOOLS = TOOL_SCHEMAS.filter((t) => EXPLORE_TOOL_NAMES.includes(t.function.name))

export function delegateChildToolSchemas(requested?: string[] | null) {
  if (!requested || !requested.length) return EXPLORE_TOOLS
  const allowed = requested.filter((name) => isReadOnlyTool(name))
  const schemas = TOOL_SCHEMAS.filter((t) => allowed.includes(t.function.name))
  return schemas.length ? schemas : EXPLORE_TOOLS
}

// Retain the parent's backend so an SSH child cannot use a local lookalike workspace.
export function auxiliaryToolContext(cwd: string, parent: Partial<ToolContext> | undefined, signal: AbortSignal): ToolContext {
  return { ...parent, cwd, signal }
}

const DEFAULT_DELEGATE_MAX_STEPS = 12

export async function runDelegateChild(
  provider: Provider,
  model: string,
  cwd: string,
  query: string,
  signal: AbortSignal,
  tools: ReturnType<typeof buildToolSchemas> = EXPLORE_TOOLS,
  maxSteps: number = DEFAULT_DELEGATE_MAX_STEPS,
  toolContext?: Partial<ToolContext>,
  assertRequestAllowed?: () => void,
): Promise<string> {
  assertThreadChildSupported(toolContext?.currentThread)
  assertRequestAllowed?.()
  const toolNames = tools.map((t) => t.function?.name).filter(Boolean).join('/') || 'list_dir/read_file/search'
  const messages: ApiMessage[] = [
    { role: 'system', content: ensureTensorFoldStudioStylePrompt(`You are Explore, a read-only codebase research specialist. Workspace: ${cwd}. Use ${toolNames} to investigate, then reply with a concise findings summary. Name the relevant files, paths, where things live, and how they connect. You CANNOT edit files or run commands.`) },
    { role: 'user', content: query },
  ]
  const cap = Math.max(1, maxSteps)
  for (let step = 0; step < cap; step++) {
    if (signal.aborted) break
    let content = ''
    let toolCalls: ToolCall[] = []
    for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages, tools, temperature: 0.2, approvalMode: 'readonly', cwd, signal, assertRequestAllowed })) {
      if (ch.type === 'content') content += ch.text
      else if (ch.type === 'tool_calls') toolCalls = ch.toolCalls
    }
    const msg: ApiMessage = { role: 'assistant', content: content || null }
    if (toolCalls.length) msg.tool_calls = toolCalls
    messages.push(msg)
    if (!toolCalls.length) return content.trim() || '(no findings)'
    for (const call of toolCalls) {
      let args: Record<string, any> = {}
      try { args = JSON.parse(call.function.arguments || '{}') } catch { args = {} }

      const matchedSchema = tools.find((t: any) => t.function?.name === call.function.name)
      if (matchedSchema?.function?.parameters) args = coerceToolArgs(args, matchedSchema.function.parameters as Record<string, unknown>)
      let result: Pick<ToolResult, 'output' | 'ok' | 'meta'>
      try { result = await executeTool(call.function.name, args, auxiliaryToolContext(cwd, toolContext, signal)) } catch (e: any) { result = { ok: false, output: `error: ${e?.message ?? e}` } }
      const view = compressToolOutput({ tool: call.function.name, ...result, args, maxChars: 8_000, maxBytes: 24_000 })
      messages.push(rememberPreparedToolOutput({ role: 'tool', tool_call_id: call.id, name: call.function.name, content: wrapUntrustedToolOutput(call.function.name, view.content) }))
    }
  }
  const last = [...messages].reverse().find((m) => m.role === 'assistant' && m.content)
  return (last?.content as string)?.trim() || '(Explore reached its step limit without a summary)'
}

export async function runExplore(provider: Provider, model: string, cwd: string, query: string, signal: AbortSignal, toolContext?: Partial<ToolContext>, assertRequestAllowed?: () => void): Promise<string> {
  return runDelegateChild(provider, model, cwd, query, signal, EXPLORE_TOOLS, 8, toolContext, assertRequestAllowed)
}

const ORACLE_MAX_STEPS = 6

export async function runOracle(provider: Provider, model: string, cwd: string, query: OracleQuery, signal: AbortSignal, toolContext?: Partial<ToolContext>, assertRequestAllowed?: () => void): Promise<string> {
  assertThreadChildSupported(toolContext?.currentThread)
  assertRequestAllowed?.()
  const messages: ApiMessage[] = [
    { role: 'system', content: ensureTensorFoldStudioStylePrompt(buildOracleSystemPrompt(cwd)) },
    { role: 'user', content: buildOracleUserMessage(query) },
  ]
  for (let step = 0; step < ORACLE_MAX_STEPS; step++) {
    if (signal.aborted) break
    let content = ''
    let toolCalls: ToolCall[] = []
    for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages, tools: EXPLORE_TOOLS, temperature: 0.2, reasoningEffort: 'high', approvalMode: 'readonly', cwd, signal, assertRequestAllowed })) {
      if (ch.type === 'content') content += ch.text
      else if (ch.type === 'tool_calls') toolCalls = ch.toolCalls
    }
    const msg: ApiMessage = { role: 'assistant', content: content || null }
    if (toolCalls.length) msg.tool_calls = toolCalls
    messages.push(msg)
    if (!toolCalls.length) return content.trim() || '(no answer)'
    for (const call of toolCalls) {
      let a: Record<string, any> = {}
      try { a = JSON.parse(call.function.arguments || '{}') } catch { a = {} }
      const matched = EXPLORE_TOOLS.find((t: any) => t.function?.name === call.function.name)
      if (matched?.function?.parameters) a = coerceToolArgs(a, matched.function.parameters as Record<string, unknown>)
      let result: Pick<ToolResult, 'output' | 'ok' | 'meta'>
      try { result = await executeTool(call.function.name, a, auxiliaryToolContext(cwd, toolContext, signal)) } catch (e: any) { result = { ok: false, output: `error: ${e?.message ?? e}` } }
      const view = compressToolOutput({ tool: call.function.name, ...result, args: a, maxChars: 8_000, maxBytes: 24_000 })
      messages.push(rememberPreparedToolOutput({ role: 'tool', tool_call_id: call.id, name: call.function.name, content: wrapUntrustedToolOutput(call.function.name, view.content) }))
    }
  }
  const last = [...messages].reverse().find((m) => m.role === 'assistant' && m.content)
  return (last?.content as string)?.trim() || '(Oracle reached its step limit without an answer)'
}

export const DELEGATE_TASK_MAX_CONCURRENT = 3

const DELEGATE_TASK_BATCH_LIMIT = 8

export function parseDelegateTaskArgs(args: Record<string, any>): DelegateTaskInput[] {
  const rawTasks = Array.isArray(args?.tasks) ? args.tasks : []
  const sharedToolNames = normalizeToolNames(args?.tool_names)
  return rawTasks
    .map((entry: any): DelegateTaskInput | null => {
      if (typeof entry === 'string') {
        const task = entry.trim()
        return task ? { task, ...(sharedToolNames ? { toolNames: sharedToolNames } : {}) } : null
      }
      if (entry && typeof entry === 'object') {
        const task = String(entry.task ?? '').trim()
        if (!task) return null
        const toolNames = normalizeToolNames(entry.tool_names) ?? sharedToolNames
        const maxSteps = Number.isFinite(Number(entry.max_steps)) ? Math.max(1, Math.floor(Number(entry.max_steps))) : undefined
        return { task, ...(toolNames ? { toolNames } : {}), ...(maxSteps ? { maxSteps } : {}) }
      }
      return null
    })
    .filter((input): input is DelegateTaskInput => input !== null)
    .slice(0, DELEGATE_TASK_BATCH_LIMIT)
}

function normalizeToolNames(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const names = value.filter((v): v is string => typeof v === 'string' && v.trim().length > 0).map((v) => v.trim())
  return names.length ? names : undefined
}

export function formatDelegateTaskResults(results: Array<{ task: string; result: string; ok: boolean; error?: string }>): string {
  if (!results.length) return '(no tasks were run)'
  return results
    .map((r, i) => {
      const header = `${i + 1}. ${r.task}`
      if (r.ok) return `${header}\n${r.result || '(no findings)'}`
      return `${header}\nFAILED: ${r.error || 'unknown error'}`
    })
    .join('\n\n')
}
