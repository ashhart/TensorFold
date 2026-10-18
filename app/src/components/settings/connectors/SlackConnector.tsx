import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { shortPath } from '../../../lib/format'
import { api } from '../../../lib/ipc'
import { Button, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import type { SlackSourceConfig } from '../../../lib/types'
import { LiveMessagingSource, type LiveMessagingSourceValue } from './LiveMessagingSource'
import { MessagingSetupGuide } from './MessagingSetupGuide'

export function SlackConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft } = ctx
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const edit = (id: string, patch: Partial<SlackSourceConfig>) => setDraft((current) => ({ ...current, slack: { ...current.slack, sources: current.slack.sources.map((source) => source.id === id ? { ...source, ...patch } : source) } }))
  const save = async (id: string, patch: Partial<SlackSourceConfig> = {}) => {
    const next = { ...draft, slack: { ...draft.slack, sources: draft.slack.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) } }
    await ctx.save(next)
  }
  const addLive = async () => {
    setBusy('add'); setError('')
    try {
      const now = Date.now()
      const source: SlackSourceConfig = { id: `slack_${crypto.randomUUID().slice(0, 8)}`, label: tx("Slack workspace"), path: '', kind: 'api_token', token: '', channels: '', enabled: false, createdAt: now, updatedAt: now, lastSyncAt: null, lastError: null }
      await ctx.save({ ...draft, slack: { ...draft.slack, sources: [source, ...draft.slack.sources] } })
    } catch { setError(tx("Could not add this connection; retry after saving any pending settings changes.")) }
    finally { setBusy('') }
  }
  const actOnExport = async (id: string, action: () => Promise<unknown>) => {
    setBusy(id); setError('')
    try { await action() } catch { setError(tx("The export action failed; check that the folder is available and try again.")) }
    finally { setBusy('') }
  }
  const addExport = async () => {
    const directory = await api.pickDirectory()
    if (!directory) return
    const next = await api.addSlackSource(directory)
    const priorIds = new Set(draft.slack.sources.map((source) => source.id))
    const added = next.sources.filter((source) => !priorIds.has(source.id))
    setDraft((current) => ({ ...current, slack: { ...current.slack, sources: [...added.filter((source) => !current.slack.sources.some((existing) => existing.id === source.id)), ...current.slack.sources] } }))
  }
  const remove = async (id: string) => { await api.removeSlackSource(id); setDraft((current) => ({ ...current, slack: { ...current.slack, sources: current.slack.sources.filter((source) => source.id !== id) } })) }
  const persistedValue = (id: string): LiveMessagingSourceValue | null | undefined => {
    if (!ctx.persistedSettings) return undefined
    const source = ctx.persistedSettings.slack.sources.find((candidate) => candidate.id === id)
    return source ? { ...source, token: source.token ?? '', channels: source.channels ?? '', credentialVersion: source.secretRotations?.token } : null
  }
  const asPatch = (patch: Partial<LiveMessagingSourceValue>): Partial<SlackSourceConfig> => {
    const { channels, ...rest } = patch
    return { ...rest, ...(channels !== undefined ? { channels } : {}) }
  }
  return <Section title={tx("Slack")} desc={tx("Connect a workspace bot or read an export folder; choose channels before enabling live access.")}>
    <div className="space-y-3">
      <MessagingSetupGuide steps={[
        'Create a Slack app for your workspace, then add Bot Token Scopes under OAuth & Permissions.',
        'For public channel history use channels:read and channels:history; private channels also need groups:read and groups:history; chat:write allows approved replies and files:read allows shared attachments.',
        'Install the app to your workspace, invite its bot to the selected channels, then save and verify its bot token here.',
      ]} links={[{ label: tx("Create Slack app"), url: 'https://api.slack.com/apps' }, { label: tx("Permission guide"), url: 'https://docs.slack.dev/authentication/tokens/' }]} />
      <p className="text-[12px] leading-relaxed text-ink-3">{tx("Slack limits public and private thread history for bot tokens, and workspace policies and rate limits still apply; channel history remains available with the scopes above.")}</p>
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card p-3.5">
        <span className="text-[13px] text-ink-2">{tx("Connections: {count}", { count: draft.slack.sources.length })}</span>
        <div className="flex flex-wrap gap-2"><Button size="sm" icon="plus" onClick={addLive} disabled={!!busy}>{busy === 'add' ? tx("Adding…") : tx("Add workspace bot")}</Button><Button size="sm" icon="folder" onClick={() => actOnExport('export', addExport)} disabled={!!busy || !!ctx.slackBusy}>{tx("Add export folder")}</Button></div>
      </div>
      {error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      {!draft.slack.sources.length && <div className="rounded-xl border border-dashed border-line p-5 text-center text-[13px] text-ink-3">{tx("Connect your first workspace bot, or choose an existing Slack export folder for local reading.")}</div>}
      {draft.slack.sources.map((source) => source.kind === 'api_token' ? <LiveMessagingSource key={source.id} brand="slack" persistedValue={persistedValue(source.id)} value={{ ...source, token: source.token ?? '', channels: source.channels ?? '', credentialVersion: source.secretRotations?.token }} edit={(patch) => edit(source.id, asPatch(patch))} save={(patch) => save(source.id, patch ? asPatch(patch) : {})} verify={async () => { const result = await api.verifySlack(source.id); return { ok: result.ok, identity: result.team ? `${result.bot || 'bot'} in ${result.team}` : result.bot, error: result.error ? 'Slack rejected this token; check that the app is installed and its token is current.' : undefined } }} listChannels={async () => (await api.listSlackChannels(source.id)).map((channel) => ({ ...channel, detail: channel.isPrivate ? 'Private channel' : 'Public channel' }))} sync={async () => { const result = await api.syncSlack(source.id); const failed = result.sources.find((candidate) => candidate.id === source.id)?.lastError; if (failed) throw new Error(tx("Slack channel read failed")); return `Read ${result.messages ?? 0} messages from this workspace.` }} remove={async () => { await remove(source.id) }} /> : <div key={source.id} className="rounded-xl border border-line bg-card p-3.5">
        <div className="flex items-start gap-3"><div className="min-w-0 flex-1"><Input aria-label={tx("Slack export label")} value={source.label} disabled={busy === source.id} onChange={(event) => edit(source.id, { label: event.target.value })} onBlur={() => actOnExport(source.id, () => save(source.id))} /><p className="mt-1 break-words text-[13px] text-ink-3">{shortPath(source.path, 5)}</p></div><Toggle label={tx("Enable {value0}", { value0: source.label || tx("Slack export") })} checked={source.enabled} disabled={!!busy} onChange={(enabled) => actOnExport(source.id, () => save(source.id, { enabled }))} /></div>
        <div className="mt-3 flex flex-wrap gap-2"><Button size="sm" disabled={!!busy || !source.enabled} onClick={() => actOnExport(source.id, async () => { await api.syncSlack(source.id) })}>{tx("Read export")}</Button><Button size="sm" variant="danger" disabled={!!busy} onClick={() => actOnExport(source.id, () => remove(source.id))}>{tx("Remove export")}</Button><span className="self-center text-[12px] text-ink-3">{source.lastSyncAt ? tx("Last read {value0}", { value0: new Date(source.lastSyncAt).toLocaleString(currentIntlLocale()) }) : tx("Not read yet")}</span></div>
        {source.lastError && <p role="alert" className="mt-2 text-[13px] text-danger">{tx("The last export read failed; check folder access and try again.")}</p>}
      </div>)}
    </div>
  </Section>
}
