import type { BotToolScope, BotThreadConfig } from '../electron/agent/types/settings'

// Child agents and alternate executors do not yet inherit a captured tool scope.
export const RESTRICTED_BOT_UNSUPPORTED_TOOLS = new Set([
  'explore', 'delegate_task', 'oracle', 'launch_subagent', 'launch_subagents',
  'subagent_status', 'subagent_send', 'subagent_inbox',
  'create_project', 'create_room', 'create_kanban_ticket', 'change_setting',
])
export const MCP_SCOPE_BRIDGES = ['tool_search', 'tool_describe', 'tool_call'] as const

export function normalizeBotToolScope(value: unknown): BotToolScope | undefined {
  if (value === undefined) return undefined
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid bot tool scope.')
  const raw = value as Record<string, unknown>
  if (Object.keys(raw).some((key) => key !== 'mode' && key !== 'names') || raw.mode !== 'selected' || !Array.isArray(raw.names) || raw.names.length > 200 || raw.names.some((name) => typeof name !== 'string' || !/^[A-Za-z0-9_-]{1,200}$/.test(name))) throw new Error('Invalid bot tool scope; select at most 200 exact tool names.')
  if (raw.names.some((name) => RESTRICTED_BOT_UNSUPPORTED_TOOLS.has(name))) throw new Error('Delegation, orchestration and global setting tools are unavailable to restricted bots.')
  return { mode: 'selected', names: [...new Set(raw.names as string[])].sort() }
}

export function isRestrictedBot(bot?: Pick<BotThreadConfig, 'toolScope' | 'toolScopeProblem'>): boolean {
  return bot?.toolScope !== undefined || bot?.toolScopeProblem !== undefined
}

export function botUsesGuardedExecution(bot?: Pick<Partial<BotThreadConfig>, 'executionTarget' | 'toolScope' | 'toolScopeProblem'>): boolean {
  return isRestrictedBot(bot) || bot?.executionTarget === 'vm'
}

/** Stored records are untrusted; display their status without granting a fallback scope. */
export function inspectBotToolScope(value: unknown): { names: string[]; error: string | null } {
  try { return { names: normalizeBotToolScope(value)?.names ?? [], error: null } }
  catch (error) { return { names: [], error: error instanceof Error ? error.message : 'Invalid bot tool scope.' } }
}

export function botToolScopeReviewText(value: unknown): string {
  try { const text = JSON.stringify(value, null, 2) ?? 'No saved selection.'; return text.length > 16_000 ? text.slice(0, 16_000) + '\n[Preview shortened; the original saved value is retained.]' : text }
  catch { return 'The saved selection cannot be displayed safely.' }
}

export function botToolScopeOriginalValue(bot: Pick<BotThreadConfig, 'toolScope' | 'toolScopeProblem'>): unknown {
  const problem = bot.toolScopeProblem
  if (problem === undefined) return bot.toolScope
  return problem && typeof problem === 'object' && Object.prototype.hasOwnProperty.call(problem, 'value') ? problem.value : problem
}

/** Add catalog transport only when an exact MCP target was selected; never grant a target. */
export function botAllowedToolNames(bot?: Pick<BotThreadConfig, 'toolScope' | 'toolScopeProblem'>, existing?: readonly string[]): string[] | undefined {
  if (bot?.toolScopeProblem !== undefined) throw new Error('Repair the saved bot tool selection before running it.')
  const scope = normalizeBotToolScope(bot?.toolScope)
  if (!scope) return existing ? [...existing] : undefined
  const names = [...scope.names.filter((name) => !MCP_SCOPE_BRIDGES.includes(name as any))]
  if (names.some((name) => name.startsWith('mcp__'))) names.push(...MCP_SCOPE_BRIDGES)
  return existing ? names.filter((name) => existing.includes(name)) : names
}

/** Used at schema, pre-hook and direct dispatch seams, including deferred MCP targets. */
export function botToolBlockReason(bot: Pick<BotThreadConfig, 'toolScope' | 'toolScopeProblem'> | undefined, name: string, args: Record<string, unknown> = {}, existing?: readonly string[]): string | null {
  let names: string[] | undefined
  try { names = botAllowedToolNames(bot, existing) } catch (error) { return error instanceof Error ? error.message : 'Invalid bot tool scope.' }
  if (names && !names.includes(name)) return "This tool is outside this bot's captured tool scope."
  if (!isRestrictedBot(bot)) return null
  if (RESTRICTED_BOT_UNSUPPORTED_TOOLS.has(name)) return 'Delegation and alternate executors are unavailable to restricted bots.'
  if (name === 'tool_call' || name === 'tool_describe') {
    const target = typeof args.name === 'string' ? args.name.trim() : ''
    if (!target.startsWith('mcp__') || !normalizeBotToolScope(bot?.toolScope)!.names.includes(target) || existing && !existing.includes(target)) return "This MCP target is outside this bot's captured tool scope."
  }
  return null
}

export function botToolScopeKey(value: unknown): string {
  return JSON.stringify(normalizeBotToolScope(value) ?? null)
}
