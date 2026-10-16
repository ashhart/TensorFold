import { botAllowedToolNames, botToolBlockReason, isRestrictedBot, normalizeBotToolScope } from '../../shared/bot-tool-policy'
import type { BotThreadConfig } from './types'
import type { ToolContext } from './tools'
import { buildCapabilityRegistry, selectedCapabilityIdsFromMessages } from './capability-registry'

/** Capture an independent scope once; a preset edit cannot change an in-flight run. */
export function scopeBotToolContext<T extends Partial<ToolContext>>(bot: BotThreadConfig | undefined, base: T): T {
  if (!isRestrictedBot(bot)) return base
  if (bot?.toolScopeProblem !== undefined) throw new Error('Repair the saved bot tool selection before running it.')
  const scope = normalizeBotToolScope(bot!.toolScope)!
  Object.freeze(scope.names); Object.freeze(scope)
  const captured = Object.freeze({ ...bot!, toolScope: scope })
  const selected = botAllowedToolNames(captured, base.allowedToolNames)!
  const registry = buildCapabilityRegistry({
    mcpTools: base.mcpTools, skills: base.skills, commands: base.commands, localScripts: base.localScripts,
    disabledToolNames: base.disabledToolNames, disabledCapabilityIds: base.disabledCapabilityIds,
    selectedCapabilityIds: [...selectedCapabilityIdsFromMessages(base.currentThread?.apiMessages ?? []), ...(base.phoneControlAvailable ? ['phone'] : [])],
    knownToolNames: selected,
  })
  const allowedToolNames = selected.filter((name) => registry.visibleToolNames.includes(name))
  Object.freeze(allowedToolNames)
  const call = base.callMcpTool
  return {
    ...base,
    currentThread: base.currentThread ? { ...base.currentThread, bot: captured } : undefined,
    allowedToolNames,
    mcpTools: (base.mcpTools ?? []).filter((tool) => allowedToolNames.includes(tool.callName)),
    packagedPlugins: [],
    callMcpTool: call ? async (name, args) => {
      const reason = botToolBlockReason(captured, name, args, allowedToolNames)
      return reason ? { ok: false, output: reason } : call(name, args)
    } : undefined,
  } as T
}
