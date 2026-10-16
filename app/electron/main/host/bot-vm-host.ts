import { BotVmService, VM_TOOL_NAMES, type BotVmOptions } from '../../agent/bot-vm/service'
import { redactSecretValues } from '../../agent/secret-detection'
import { SECRET_MASK } from '../../agent/security'
import type { ToolContext } from '../../agent/tools'

export { VM_TOOL_NAMES }
export type BotVmHost = BotVmService
export function createBotVmHost(options: BotVmOptions): BotVmHost { return new BotVmService(options) }
let host: BotVmHost | null = null
export function initializeBotVmHost(options: BotVmOptions): BotVmHost { return host ??= createBotVmHost(options) }
export function getBotVmHost(): BotVmHost | null { return host }

/** Retain only the chat identity and cancellation; never retain host callbacks,
 * plugins, connections, file backends or managed-output access in a VM run. */
export function createBotVmToolContext(vm: BotVmHost, botId: string, base: ToolContext): ToolContext {
  return {
    cwd: '/workspace', signal: base.signal, currentThread: base.currentThread,
    workspace: null, allowedToolNames: [...VM_TOOL_NAMES, 'ask_user'],
    networkCapabilityAllowed: false, sensitiveRetrievalAllowed: false,
    allowSandboxNetwork: false, mcpTools: [], skills: [], commands: [], localScripts: [], packagedPlugins: [],
    botVm: { executeTool: async (name, args, signal) => {
      const result = await vm.executeTool(botId, name, args, signal)
      const redacted = redactSecretValues(result.output, SECRET_MASK)
      return { ...result, output: redacted.text, title: name === 'run_command' ? 'Run in bot VM' : String(args.path ?? '/workspace'), command: name === 'run_command' ? String(args.command ?? '') : undefined, meta: `isolated Linux VM · network disabled${result.code != null ? ` · exit ${result.code}` : ''}${result.truncated ? ' · bounded' : ''}${redacted.hits.length ? ` · redacted=${redacted.hits.length}` : ''}` }
    } },
  }
}
