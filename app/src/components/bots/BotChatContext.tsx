import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useStore } from '../../lib/store'
import type { Thread } from '../../lib/types'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { botToolScopeOriginalValue, botToolScopeReviewText, botUsesGuardedExecution, inspectBotToolScope } from '../../../shared/bot-tool-policy'

export function BotChatContext({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const preset = useStore((state) => state.settings?.agentPresets.find((bot) => bot.id === thread.bot?.presetId))
  const bot = thread.bot
  if (!bot) return null
  const scope = inspectBotToolScope(bot.toolScope)
  const invalidScope = !!scope.error || bot.toolScopeProblem !== undefined || preset?.toolScopeProblem !== undefined
  const blocked = invalidScope ? tx("This bot tool selection needs review.") : !preset || preset.archivedAt ? tx("This bot is archived.") : preset.paused ? tx("This bot is paused.") : null
  const empty = !thread.timeline.some((item) => item.kind === 'user')
  if (!empty && !blocked) return null
  return <section aria-label={tx("About this bot chat")} className="rounded-xl border border-line bg-card p-4">
    {blocked ? <><p className="text-[14px] font-medium text-ink">{blocked}</p><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{invalidScope ? tx("Your conversation is retained. Open Bots to review the selection and start a new chat; this captured configuration stays blocked.") : tx("Your conversation is still here. Restore or resume the bot before continuing.")}</p>{invalidScope && <details className="mt-3 text-[12px] text-ink-2"><summary className="cursor-pointer">{tx("Original captured selection")}</summary><pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-all">{botToolScopeReviewText(botToolScopeOriginalValue(bot))}</pre></details>}<Button className="mt-3" size="sm" onClick={() => useStore.getState().setView({ name: 'bots' })}>{tx("Manage bot")}</Button></> : <>
      <Icon name="spark" size={20} className="mb-3 text-accent" />
      <h2 className="break-words text-[17px] font-semibold text-ink">{tx("Tell ")}{bot.name} {tx("what you need")}</h2>
      {bot.role && <p className="mt-1 break-words text-[13px] leading-relaxed text-ink-2">{bot.role}</p>}
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{bot.approvalScope === 'readonly' ? tx("Read only keeps this bot from taking actions.") : tx("Actions follow your saved workspace approval rules.")} {bot.executionTarget === 'vm' ? tx("Files and commands run in this bot’s separate VM with guest network off.") : tx("This chat uses your configured tools and sources on this Mac.")} {tx("You can stop a run at any time.")}</p>
      {bot.toolScope && <p className="mt-2 break-words text-[12px] leading-relaxed text-ink-2">{tx("Captured tool access: ")}{scope.names.join(', ') || tx("no tools")}{tx(". Plugins and delegation are unavailable; saved approvals and device grants still apply.")}</p>}
      {botUsesGuardedExecution(bot) && <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Compaction uses a local summary without extra model requests. Automatic titles and skill review are disabled.")}</p>}
      <details className="mt-3 text-[12px] text-ink-2"><summary className="cursor-pointer text-ink-2">{tx("Captured instructions")}</summary><p className="mt-2 whitespace-pre-wrap break-words leading-relaxed">{bot.instructions || tx("No extra instructions were added.")}</p><p className="mt-2 leading-relaxed">{tx("Editing the bot changes new chats; this conversation keeps its captured configuration.")}</p></details>
    </>}
  </section>
}
