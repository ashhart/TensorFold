import path from 'node:path'
import { formatBrowserAutomationResult, normalizeBrowserAutomationInput } from './browser-automation'
import { normalizeComputerUseInput } from './computer-use/actions'
import { normalizePhoneAction } from './phone/service'
import type { PhoneToolInput } from './types'
import { formatControllerActionResult, normalizeControllerActionInput } from './controller-actions'
import type { SubAgentMessageKind } from './types'
import type { ToolContext, ToolResult } from './tool-context'


// Device control and child-agent actions retain explicit availability and privacy guards.
export async function executeControlTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'browser_automation': {
      const input = normalizeBrowserAutomationInput(args)
      if (!ctx.runBrowserAutomation) return { ok: false, output: 'Browser automation runner is not available in this environment.' }
      const result = await ctx.runBrowserAutomation(input)
      return {
        ok: result.ok,
        output: formatBrowserAutomationResult(result),
        title: `Browser ${result.action}`,
        path: result.screenshot?.path,
        meta: `session=${result.sessionId}; tab=${result.tabId}${result.loadState ? `; load=${result.loadState}` : ''}`,
      }
    }

    case 'controller_action': {
      const input = normalizeControllerActionInput(args, { ownAppName: 'TensorFold' })
      if (!ctx.runControllerAction) return { ok: false, output: 'Controller action runner is not available in this environment.' }
      const result = await ctx.runControllerAction(input)
      return {
        ok: result.ok,
        output: formatControllerActionResult(result),
        title: `Controller ${result.action}`,
        meta: `${result.appName} · ${result.target}`,
      }
    }

    case 'computer_use': {
      const input = normalizeComputerUseInput(args)
      if (!ctx.runComputerUse) return { ok: false, output: 'Computer use is not available in this environment.' }
      const result = await ctx.runComputerUse(input)
      return { ok: result.ok, output: result.output, title: result.title, meta: result.meta }
    }

    case 'phone_observe':
    case 'phone_control': {
      if (!ctx.phoneControlAvailable || !ctx.runPhoneControl) return { ok: false, title: 'Phone control', output: 'Open Phone in this chat, connect your device, select a local model and choose Allow this chat first.' }
      if (!ctx.phonePrivacyGuard) return { ok: false, title: 'Phone control', output: 'The local phone privacy guard is unavailable; no phone content or action was requested.' }
      const privacyReason = ctx.phonePrivacyGuard()
      if (privacyReason) return { ok: false, title: 'Phone control', output: privacyReason }
      const input: PhoneToolInput = name === 'phone_observe' ? { action: 'observe', ...(typeof args.sessionId === 'string' ? { sessionId: args.sessionId } : {}) } : normalizePhoneAction(args)
      return ctx.runPhoneControl(input)
    }

    case 'launch_subagent': {
      if (!ctx.launchSubAgent) return { ok: false, output: 'Sub-agent launcher is not available in this run.' }
      const task = String(args.task ?? '').trim()
      if (!task) return { ok: false, output: 'launch_subagent requires a task.' }
      const launched = await ctx.launchSubAgent({
        task,
        name: typeof args.name === 'string' ? args.name : null,
        model: typeof args.model === 'string' ? args.model : null,
        providerId: typeof args.providerId === 'string' ? args.providerId : null,
        accountId: typeof args.accountId === 'string' ? args.accountId : null,
        agent: typeof args.agent === 'string' ? args.agent : null,
        background: args.background === true,
      })
      return {
        ok: launched.agent.status !== 'failed' && launched.agent.status !== 'cancelled',
        output: launched.output,
        title: launched.agent.name,
        path: launched.agent.parent.label,
        meta: `sub-agent ${launched.agent.status}`,
        subAgent: launched.agent,
      }
    }

    case 'launch_subagents': {
      if (!ctx.launchSubAgent) return { ok: false, output: 'Sub-agent launcher is not available in this run.' }
      const launchSubAgent = ctx.launchSubAgent
      const rawTasks = Array.isArray(args.tasks) ? args.tasks : []
      const tasks = rawTasks
        .map((item) => typeof item === 'string'
          ? { task: item, name: '', model: '', providerId: '', accountId: '', agent: '', background: undefined as boolean | undefined }
          : {
            task: String(item?.task ?? '').trim(),
            name: String(item?.name ?? '').trim(),
            model: String(item?.model ?? '').trim(),
            providerId: String(item?.providerId ?? '').trim(),
            accountId: String(item?.accountId ?? '').trim(),
            agent: typeof item?.agent === 'string' ? item.agent.trim() : (typeof args.agent === 'string' ? args.agent.trim() : ''),
            background: typeof item?.background === 'boolean' ? item.background : undefined,
          })
        .filter((item) => item.task)
        .slice(0, 3)
      if (!tasks.length) return { ok: false, output: 'launch_subagents requires at least one task.' }

      const results = await Promise.all(tasks.map(async (task, index) => {
        try {
          const launched = await launchSubAgent({
            task: task.task,
            name: task.name || null,
            model: task.model || null,
            providerId: task.providerId || null,
            accountId: task.accountId || null,
            agent: task.agent || null,
            background: task.background ?? (args.background === true),
          })
          const label = launched.agent.name || task.name || `Sub-agent ${index + 1}`
          return {
            ok: launched.agent.status !== 'failed' && launched.agent.status !== 'cancelled',
            output: `${index + 1}. ${label}: ${launched.output}`,
          }
        } catch (error: any) {
          return { ok: false, output: `${index + 1}. ${task.name || `Sub-agent ${index + 1}`}: failed: ${error?.message ?? error}` }
        }
      }))
      const omitted = rawTasks.length > tasks.length ? `\n\n${rawTasks.length - tasks.length} extra task(s) were not launched; batch limit is 3.` : ''
      return {
        ok: results.every((result) => result.ok),
        output: results.map((result) => result.output).join('\n') + omitted,
        title: 'Sub-agent batch',
        meta: `${tasks.length} launched${rawTasks.length > tasks.length ? ` · ${rawTasks.length - tasks.length} omitted` : ''}`,
      }
    }

    case 'subagent_status': {
      const ids = Array.isArray(args.agent_ids)
        ? new Set(args.agent_ids.map((value: unknown) => String(value)).filter(Boolean))
        : null
      const currentThreadId = ctx.currentThread?.id
      const agents = (ctx.listSubAgents?.() ?? [])
        .filter((agent) => (!!ctx.currentSubAgentId && agent.id === ctx.currentSubAgentId) || [agent.threadId, agent.parent?.threadId].filter(Boolean).every((id) => id === currentThreadId || ctx.threads?.some((thread) => thread.id === id)))
        .filter((agent) => ids?.size ? ids.has(agent.id) : !currentThreadId || agent.threadId === currentThreadId)
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, 30)
      if (!agents.length) return { ok: true, output: 'No matching sub-agents.', title: 'Sub-agent status', meta: '0 agents' }
      ctx.recordRecallSources?.([...new Set(agents.flatMap((agent) => [agent.threadId, agent.parent?.threadId]).filter((id): id is string => !!id))])
      ctx.assertRecallSourcesAllowed?.()
      const output = agents.map((agent, index) => {
        const unread = agent.messages.filter((message) =>
          (message.audience === 'parent' || message.audience === 'both') && message.readByParentAt === null,
        )
        const questions = unread.filter((message) => message.kind === 'question').length
        return [
          `${index + 1}. ${agent.name} [${agent.status}]`,
          `   id: ${agent.id}`,
          `   task: ${agent.task}`,
          unread.length ? `   attention: ${unread.length} unread${questions ? ` · ${questions} question${questions === 1 ? '' : 's'}` : ''}` : '',
          agent.result ? `   result: ${agent.result.slice(0, 3000)}` : '',
          agent.error ? `   error: ${agent.error.slice(0, 1000)}` : '',
        ].filter(Boolean).join('\n')
      }).join('\n\n')
      return { ok: true, output, title: 'Sub-agent status', meta: `${agents.length} agents` }
    }

    case 'subagent_send': {
      if (!ctx.sendSubAgentMessage) return { ok: false, output: 'Sub-agent mailbox is not available in this run.' }
      const agentId = String(args.agent_id ?? ctx.currentSubAgentId ?? '').trim()
      const text = String(args.text ?? '').trim()
      if (!agentId || !text) return { ok: false, output: 'subagent_send requires an agent id and message text.' }
      if (ctx.currentSubAgentId && agentId !== ctx.currentSubAgentId) {
        return { ok: false, output: 'A child sub-agent may only write to its own mailbox.' }
      }
      const kind = typeof args.kind === 'string' ? args.kind as SubAgentMessageKind : undefined
      const message = ctx.sendSubAgentMessage({
        agentId,
        text,
        kind,
        replyToId: typeof args.reply_to_id === 'string' ? args.reply_to_id : null,
      })
      return {
        ok: true,
        output: `Mailbox ${message.kind} sent. id=${message.id}`,
        title: 'Sub-agent message',
        path: agentId,
        meta: message.kind,
      }
    }

    case 'subagent_inbox': {
      if (!ctx.readSubAgentMessages) return { ok: false, output: 'Sub-agent mailbox is not available in this run.' }
      const agentId = String(args.agent_id ?? ctx.currentSubAgentId ?? '').trim()
      if (!agentId) return { ok: false, output: 'subagent_inbox requires an agent id.' }
      if (ctx.currentSubAgentId && agentId !== ctx.currentSubAgentId) {
        return { ok: false, output: 'A child sub-agent may only read its own mailbox.' }
      }
      const messages = ctx.readSubAgentMessages({
        agentId,
        unreadOnly: args.unread_only !== false,
        markRead: args.mark_read !== false,
      })
      const output = messages.length
        ? messages.map((message) => [
          `[${message.id}] ${message.kind} from ${message.sender}${message.replyToId ? ` (reply to ${message.replyToId})` : ''}`,
          message.text,
        ].join('\n')).join('\n\n')
        : 'No matching mailbox messages.'
      return { ok: true, output, title: 'Sub-agent inbox', path: agentId, meta: `${messages.length} messages` }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
