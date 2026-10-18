import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { api } from '../../../lib/ipc'
import { Button } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import type { DiscordSource } from '../../../lib/types'
import { LiveMessagingSource, type LiveMessagingSourceValue } from './LiveMessagingSource'
import { MessagingSetupGuide } from './MessagingSetupGuide'

export function DiscordConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft } = ctx
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const persistedValue = (id: string): LiveMessagingSourceValue | null | undefined => {
    if (!ctx.persistedSettings) return undefined
    const source = ctx.persistedSettings.discord.sources.find((candidate) => candidate.id === id)
    return source ? { ...source, channels: source.allowedChannels, credentialVersion: source.secretRotations?.token } : null
  }
  const asPatch = (patch: Partial<LiveMessagingSourceValue>): Partial<DiscordSource> => {
    const { channels, ...rest } = patch
    return { ...rest, ...(channels !== undefined ? { allowedChannels: channels } : {}) }
  }
  const edit = (id: string, patch: Partial<DiscordSource>) => setDraft((current) => ({ ...current, discord: { ...current.discord, sources: current.discord.sources.map((source) => source.id === id ? { ...source, ...patch } : source) } }))
  const save = async (id: string, patch: Partial<DiscordSource> = {}) => {
    await ctx.save({ ...draft, discord: { ...draft.discord, sources: draft.discord.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) } })
  }
  const add = async () => {
    setBusy(true); setError('')
    try {
      const next = await api.addDiscordSource('Discord server')
      const priorIds = new Set(draft.discord.sources.map((source) => source.id))
      const added = next.sources.filter((source) => !priorIds.has(source.id))
      setDraft((current) => ({ ...current, discord: { ...current.discord, sources: [...current.discord.sources, ...added.filter((source) => !current.discord.sources.some((existing) => existing.id === source.id))] } }))
    }
    catch { setError(tx("Could not add this bot; retry after saving any pending settings changes.")) }
    finally { setBusy(false) }
  }
  const remove = async (id: string) => { await api.removeDiscordSource(id); setDraft((current) => ({ ...current, discord: { ...current.discord, sources: current.discord.sources.filter((source) => source.id !== id) } })) }
  return <Section title={tx("Discord")} desc={tx("Connect a server bot, verify it, then choose the exact channels TensorFold may read and reply in.")}>
    <div className="space-y-3">
      <MessagingSetupGuide steps={[
        'Create an application in the Discord Developer Portal and copy its bot token from the Bot page; personal user tokens are unsupported.',
        'Enable Message Content Intent on the Bot page when you need ordinary channel message text; Discord may require approval for eligible verified apps.',
        'Install the bot to your server with the bot scope and only View Channels, Read Message History and Send Messages; add Send Messages in Threads only when needed, without Administrator.',
      ]} links={[{ label: tx("Discord Developer Portal"), url: 'https://discord.com/developers/applications' }, { label: tx("Bot permissions"), url: 'https://docs.discord.com/developers/topics/permissions' }]} />
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card p-3.5"><span className="text-[13px] text-ink-2">{tx("Connections: {count}", { count: draft.discord.sources.length })}</span><Button size="sm" icon="plus" onClick={add} disabled={busy}>{busy ? tx("Adding…") : tx("Add server bot")}</Button></div>
      {error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      {!draft.discord.sources.length && <div className="rounded-xl border border-dashed border-line p-5 text-center text-[13px] text-ink-3">{tx("Add a server bot, then verify its token and select channels; nothing is read until you connect and request a sync.")}</div>}
      {draft.discord.sources.map((source) => <LiveMessagingSource key={source.id} brand="discord" persistedValue={persistedValue(source.id)} value={{ ...source, channels: source.allowedChannels, credentialVersion: source.secretRotations?.token }} edit={(patch) => edit(source.id, asPatch(patch))} save={(patch) => save(source.id, patch ? asPatch(patch) : {})} verify={async () => { const result = await api.verifyDiscord(source.id); return { ok: result.ok, identity: result.bot ? `${result.bot}${result.guilds?.length ? ` in ${result.guilds.length} server${result.guilds.length === 1 ? '' : 's'}` : ''}` : undefined, error: result.error ? 'Discord rejected this bot token; check the token and server installation, then retry.' : undefined } }} listChannels={async () => (await api.listDiscordChannels(source.id)).map((channel) => ({ id: channel.id, name: channel.name, detail: `${channel.guildName}${channel.threadId ? ' · Thread' : ''}` }))} sync={async () => { const result = await api.syncDiscord(source.id); return `Read ${result.messages} messages; ${result.events} new inbox items from this connection.` }} remove={() => remove(source.id)} />)}
    </div>
  </Section>
}
