import { useMemo, useState } from 'react'
import clsx from 'clsx'
import { botProviderSupported, captureBotConfig } from '../../shared/bot-policy'
import { isExternalProvider } from '../../shared/provider-trust'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { modelDisplayName, providerModelGroups } from '../lib/provider-accounts'
import { useStore } from '../lib/store'
import { mostRecentCompatibleBotChat } from '../lib/bot-chats'
import type { AgentPreset, Provider } from '../lib/types'
import { Button, Input } from './ui'
import { BotVmControls } from './bots/BotVmControls'
import { BotRecipes } from './bots/BotRecipes'
import { BotToolScopeEditor } from './bots/BotToolScopeEditor'
import { inspectBotToolScope } from '../../shared/bot-tool-policy'

const EMPTY_BOTS: AgentPreset[] = []
const EMPTY_PROVIDERS: Provider[] = []
const templates = [
  { name: 'Writing partner', role: 'Draft and refine clear writing', instructions: 'Help me write clearly. Ask about audience and purpose when they are unclear. Preserve facts and my voice. Offer a finished draft and explain only consequential edits.', lane: 'planning' as const },
  { name: 'Research helper', role: 'Find evidence and explain what it supports', instructions: 'Investigate the question using available sources. Distinguish verified facts from inference. Cite sources, identify uncertainty and ask before sending information outside this workspace.', lane: 'research' as const },
  { name: 'Code reviewer', role: 'Find concrete correctness and security issues', instructions: 'Review the code I provide. Prioritize reproducible defects, security boundaries and missing error handling. Explain the trigger and consequence of each finding. Do not invent defects or change files without my request.', lane: 'review' as const },
]

function choiceValue(providerId: string, accountId: string | null | undefined, model: string): string {
  return JSON.stringify([providerId, accountId ?? null, model])
}

function botProblem(bot: AgentPreset, providers: Provider[]): string | null {
  try { captureBotConfig({ ...bot, paused: false, archivedAt: null }, providers); return null }
  catch (error) { return error instanceof Error ? error.message : 'Check this bot’s configuration.' }
}

export function Bots() {
  const settings = useStore((state) => state.settings)
  const presets = settings?.agentPresets ?? EMPTY_BOTS
  const providers = settings?.providers ?? EMPTY_PROVIDERS
  const threads = useStore((state) => state.threads)
  const statuses = useStore((state) => state.status)
  const [query, setQuery] = useState('')
  const [showArchived, setShowArchived] = useState(false)
  const [draft, setDraft] = useState<AgentPreset | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const models = useMemo(() => [...providers].filter(botProviderSupported).sort((a, b) => Number(b.source === 'tensorfold') - Number(a.source === 'tensorfold') || Number(b.kind === 'local') - Number(a.kind === 'local')).flatMap((provider) => providerModelGroups(provider).filter((group) => group.models.length).map((group) => ({ provider, group }))), [providers])
  const visible = presets.filter((bot) => showArchived ? !!bot.archivedAt : !bot.archivedAt).filter((bot) => `${bot.name} ${bot.role}`.toLowerCase().includes(query.toLowerCase().trim()))
  const draftProblem = draft ? botProblem(draft, providers) : null
  const draftChoice = draft ? choiceValue(draft.providerId, draft.providerAccountId, draft.model) : ''
  const draftModels = draft?.executionTarget === 'vm' ? models.filter(({ provider }) => !isExternalProvider(provider)) : models
  const draftModelMissing = !!draft && !draftModels.some(({ provider, group }) => group.models.some((model) => choiceValue(provider.id, group.accountId, model) === draftChoice))

  const startDraft = (template = templates[0]) => {
    if (draft) { setError('Save or discard the current draft first.'); return }
    const first = models[0]
    const now = Date.now()
    setError(null)
    setDraft({ id: `bot_${crypto.randomUUID()}`, name: template.name, role: template.role, instructions: template.instructions, lane: template.lane, providerId: first?.provider.id ?? '', providerAccountId: first?.group.accountId ?? null, model: first?.group.models[0] ?? '', approvalScope: 'readonly', executionTarget: 'workspace', paused: false, archivedAt: null, createdAt: now, updatedAt: now })
  }
  const saveBot = async (bot: AgentPreset) => {
    setBusyId(bot.id); setError(null)
    try {
      const current = useStore.getState().settings
      if (!current) throw new Error('Settings are still loading.')
      const next = { ...bot, name: bot.name.trim(), role: bot.role.trim(), updatedAt: Date.now() }
      // A paused or archived bot may keep a temporarily missing model, allowing recovery.
      if (!next.paused && !next.archivedAt) captureBotConfig(next, current.providers)
      await useStore.getState().saveSettings({ ...current, agentPresets: current.agentPresets.some((item) => item.id === next.id) ? current.agentPresets.map((item) => item.id === next.id ? next : item) : [next, ...current.agentPresets] }, { silent: true })
      if (draft?.id === bot.id) setDraft(null)
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save the bot. Your draft is still here.') }
    finally { setBusyId(null) }
  }
  const chat = async (bot: AgentPreset) => {
    setBusyId(bot.id); setError(null)
    try {
      const thread = await api.createBotChat(bot.id)
      useStore.setState((state) => ({ threads: [thread, ...state.threads.filter((item) => item.id !== thread.id)], view: { name: 'thread', id: thread.id } }))
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start the bot chat.') }
    finally { setBusyId(null) }
  }

  return <div className="scroll-thin h-full overflow-y-auto px-4 py-6 sm:px-7">
    <div className="mx-auto max-w-[1080px]">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div><h1 className="text-[22px] font-semibold tracking-[-0.02em] text-ink">Your bots</h1><p className="mt-1 max-w-[620px] text-[13px] leading-relaxed text-ink-2">Save a model, a responsibility and clear permissions, then start a chat. Bots work while TensorFold is running on this Mac.</p></div>
        <Button variant="primary" icon="plus" disabled={!!draft || !!busyId} onClick={() => startDraft()}>New bot</Button>
      </div>
      <BotRecipes disabled={!!draft || !!busyId} />
      {api.platform === 'web-remote' && <div className="mb-4 rounded-lg border border-line bg-card p-3 text-[13px] text-ink-2">Manage bots in the desktop app on the host Mac.</div>}
      {error && <div role="alert" className="mb-4 flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/5 p-3 text-[13px] text-danger"><Icon name="alert" size={16} className="mt-0.5 shrink-0" /><span className="min-w-0 flex-1 break-words">{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}><Icon name="x" size={15} /></button></div>}
      {draft ? <section aria-label="Bot editor" className="mb-6 overflow-hidden rounded-xl border border-line-strong bg-card shadow-card">
        <div className="border-b border-line px-4 py-3 text-[14px] font-semibold">{presets.some((bot) => bot.id === draft.id) ? 'Edit bot' : 'Create a bot'}</div>
        <form onSubmit={(event) => { event.preventDefault(); void saveBot(draftProblem ? { ...draft, paused: true } : draft) }} className="space-y-4 p-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Name</span><Input autoFocus required maxLength={80} value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} placeholder="Writing partner" /></label>
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Responsibility</span><Input maxLength={240} value={draft.role} onChange={(event) => setDraft({ ...draft, role: event.target.value })} placeholder="What should this bot help with?" /></label>
          </div>
          <label className="block space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Instructions</span><textarea rows={5} maxLength={16_000} value={draft.instructions ?? ''} onChange={(event) => setDraft({ ...draft, instructions: event.target.value })} className="w-full resize-y rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-relaxed text-ink outline-none focus:border-accent" placeholder="Describe its job, what a good answer looks like, and when it should ask you." /><span className="block text-[12px] text-ink-2">Keep passwords and API keys in Settings. Instructions cannot grant permissions.</span></label>
          <label className="block space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Model</span><select required value={draftChoice} onChange={(event) => { const [providerId, providerAccountId, model] = JSON.parse(event.target.value); setDraft({ ...draft, providerId, providerAccountId, model }) }} className="h-10 w-full min-w-0 rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink">
            {draftModelMissing && <option value={draftChoice}>{draft.model ? `Unavailable: ${draft.model}` : 'Choose a model source in Settings'}</option>}
            {draftModels.map(({ provider, group }) => <optgroup key={`${provider.id}:${group.id}`} label={`${isExternalProvider(provider) ? 'External' : 'Local'} · ${group.label}`}>{group.models.map((model) => <option key={model} value={choiceValue(provider.id, group.accountId, model)}>{modelDisplayName(provider, model)}</option>)}</optgroup>)}
          </select><span className="block text-[12px] text-ink-2">{draft.executionTarget === 'vm' ? 'A bot VM needs a model on this Mac with Cortheon off. Guest files stay out of external model services.' : 'API models keep permissions under TensorFold control. External models follow the app’s planning policy.'}</span></label>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Permissions</span><select value={draft.approvalScope ?? 'readonly'} onChange={(event) => setDraft({ ...draft, approvalScope: event.target.value as AgentPreset['approvalScope'] })} className="h-10 w-full rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink"><option value="readonly">Read only</option><option value="review">Review actions with workspace rules</option></select></label>
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">Execution</span><select value={draft.executionTarget ?? 'workspace'} onChange={(event) => setDraft({ ...draft, executionTarget: event.target.value as AgentPreset['executionTarget'] })} className="h-10 w-full rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink"><option value="workspace">This Mac</option><option value="vm">Separate Linux VM</option></select></label>
          </div>
          <BotToolScopeEditor bot={draft} onChange={setDraft} />
          <p className="rounded-lg bg-card-2 p-3 text-[12px] leading-relaxed text-ink-2">{draft.executionTarget === 'vm' ? 'A separate VM keeps bot files and commands away from your Mac. Set it up explicitly before the first run; TensorFold refuses to fall back to your Mac.' : 'This Mac uses your configured chat tools and connected sources. Read only prevents actions; workspace rules may include approvals you saved earlier.'} Changes apply to new chats; existing chats keep their captured instructions, model and permissions.</p>
          {draftProblem && <p className="text-[13px] text-ink-2">{draftProblem} Save paused to keep these instructions while you set up a model in Settings.</p>}
          <div className="flex justify-end gap-2"><Button disabled={!!busyId} onClick={() => { setDraft(null); setError(null) }}>Discard</Button><Button type="button" variant="primary" onClick={() => void saveBot(draftProblem ? { ...draft, paused: true } : draft)} disabled={!!busyId || !draft.name.trim() || api.platform === 'web-remote'}>{busyId ? 'Saving…' : draftProblem ? 'Save paused' : 'Save bot'}</Button></div>
        </form>
      </section> : null}
      <div className="mb-4 flex flex-wrap items-center gap-3"><label className="flex min-w-[180px] flex-1 items-center gap-2 rounded-lg border border-line bg-card px-3"><Icon name="search" size={15} className="text-ink-2" /><input aria-label="Search bots" placeholder="Search your bots" value={query} onChange={(event) => setQuery(event.target.value)} className="h-9 min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none" /></label><button className={clsx('rounded-lg border px-3 py-2 text-[13px]', showArchived ? 'border-line-strong bg-card text-ink' : 'border-line text-ink-2')} aria-pressed={showArchived} onClick={() => setShowArchived(!showArchived)}>Archived</button></div>
      {!visible.length ? <div className="rounded-xl border border-dashed border-line px-4 py-9 text-center"><Icon name="spark" size={25} className="mx-auto text-ink-2" /><h2 className="mt-3 text-[15px] font-medium">{query ? 'No matching bots' : showArchived ? 'No archived bots' : 'Make one bot useful first'}</h2><p className="mx-auto mt-1 max-w-[430px] text-[13px] leading-relaxed text-ink-2">{query ? 'Try a different name or responsibility.' : showArchived ? 'Archiving keeps chats and VM files so you can restore the bot later.' : 'Choose a starting point and adjust its instructions. Each bot gets its own chat history and clear permission limit.'}</p>{!query && !showArchived && <div className="mt-5 flex flex-wrap justify-center gap-2">{templates.map((template) => <Button key={template.name} size="sm" disabled={!!draft} onClick={() => startDraft(template)}>{template.name}</Button>)}</div>}</div> : <div className="grid gap-4 lg:grid-cols-2">{visible.map((bot) => {
        const history = threads.filter((thread) => thread.bot?.presetId === bot.id && !thread.archivedAt).sort((a, b) => b.updatedAt - a.updatedAt)
        const compatible = mostRecentCompatibleBotChat(bot, providers, history)
        const running = history.find((thread) => ['thinking', 'streaming', 'tool', 'awaiting'].includes(statuses[thread.id] ?? '') || thread.status === 'running')
        const awaiting = running && statuses[running.id] === 'awaiting'
        const problem = botProblem(bot, providers)
        const toolSelection = inspectBotToolScope(bot.toolScope)
        const provider = providers.find((item) => item.id === bot.providerId)
        const state = bot.archivedAt ? 'Archived' : bot.paused ? 'Paused' : awaiting ? 'Needs you' : running ? 'Working' : problem ? 'Setup needed' : bot.executionTarget === 'vm' ? 'Model ready' : 'Ready'
        return <article key={bot.id} className="min-w-0 rounded-xl border border-line bg-card p-4 shadow-card">
          <div className="flex items-start gap-3"><span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-ink-5 text-accent"><Icon name="spark" size={19} /></span><div className="min-w-0 flex-1"><h2 className="break-words text-[15px] font-semibold">{bot.name}</h2><p className="mt-0.5 break-words text-[13px] text-ink-2">{bot.role || 'A bot with your saved instructions'}</p></div><span className={clsx('shrink-0 rounded-full px-2 py-1 text-[12px]', awaiting ? 'bg-accent/10 text-accent' : 'bg-card-2 text-ink-2')}>{state}</span></div>
          <div className="mt-4 space-y-1 text-[12px] text-ink-2"><p className="truncate" title={bot.model}>{modelDisplayName(provider, bot.model)} · {isExternalProvider(provider) ? 'Planning only' : 'Local model'}</p><p>{bot.executionTarget === 'vm' ? 'Separate Linux VM' : 'This Mac'} · {bot.approvalScope === 'review' ? 'Workspace approval rules' : 'Read only'}</p></div>
          {(bot.toolScope !== undefined || bot.toolScopeProblem !== undefined) && <p className="mt-1 text-[12px] text-ink-2">{bot.toolScopeProblem !== undefined || toolSelection.error ? 'Tool selection needs repair' : toolSelection.names.length ? `${toolSelection.names.length} selected tool names` : 'Chat without tools'} · plugins and delegation off</p>}
          {problem && <p className="mt-3 break-words text-[12px] text-danger">{problem}</p>}
          {bot.executionTarget === 'vm' && <BotVmControls botId={bot.id} archived={!!bot.archivedAt} paused={!!bot.paused} />}
          <div className="mt-4 flex flex-wrap gap-2">{compatible && <Button variant="primary" size="sm" icon="message" onClick={() => useStore.getState().openThread(compatible.id)}>Open chat</Button>}{!bot.archivedAt && <Button variant={compatible ? 'default' : 'primary'} size="sm" icon={compatible ? 'plus' : 'message'} onClick={() => void chat(bot)} disabled={!!busyId || !!bot.paused || !!problem || api.platform === 'web-remote'}>{busyId === bot.id ? 'Starting…' : 'New chat'}</Button>}<Button size="sm" variant="ghost" disabled={!!draft || !!busyId} onClick={() => { setError(null); setDraft({ ...bot }) }}>Edit</Button>{!bot.archivedAt && <Button size="sm" variant="ghost" disabled={!!busyId} onClick={() => void saveBot({ ...bot, paused: !bot.paused })}>{bot.paused ? 'Resume' : running ? 'Pause and stop' : 'Pause'}</Button>}<Button size="sm" variant="ghost" disabled={!!busyId} onClick={() => void saveBot({ ...bot, archivedAt: bot.archivedAt ? null : Date.now() })}>{bot.archivedAt ? 'Restore' : 'Archive'}</Button></div>
          {!!history.length && <div className="mt-4 border-t border-line pt-3"><div className="mb-1 text-[12px] font-medium text-ink-2">Recent chats</div>{history.slice(0, 3).map((thread) => <button key={thread.id} onClick={() => useStore.getState().openThread(thread.id)} className="flex w-full items-center gap-2 rounded-md py-1.5 text-left text-[12px] text-ink-2 hover:bg-card-2"><Icon name="message" size={12} className="shrink-0 text-ink-2" /><span className="min-w-0 flex-1 truncate">{thread.title}</span><span className="shrink-0 text-ink-2">{statuses[thread.id] === 'awaiting' ? 'Needs you' : thread.status === 'running' ? 'Working' : thread.status === 'error' ? 'Error' : 'Open'}</span></button>)}</div>}
        </article>
      })}</div>}
    </div>
  </div>
}
