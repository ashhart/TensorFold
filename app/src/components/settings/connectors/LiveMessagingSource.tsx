import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { BrandIcon } from '../../../lib/brand'
import { Button, Field, Input } from '../../ui'
import { createMessagingConnectionState, messagingChannelEntries, messagingChannelScopeError, type MessagingActionState, type MessagingVerification } from './messaging-connection-state'
import { liveMessagingDraftKey, useMessagingDraftState } from './useMessagingDraftState'

export interface MessagingChannelChoice { id: string; name: string; detail?: string }
export interface LiveMessagingSourceValue {
  id: string; label: string; token: string; channels: string; enabled: boolean
  channelPrompt?: string; lastSyncAt: number | null; lastError?: string | null
  credentialVersion?: number | null
}
export function LiveMessagingSource({ brand, value, persistedValue, edit, save, verify, listChannels, sync, remove }: {
  brand: 'slack' | 'discord'
  value: LiveMessagingSourceValue
  persistedValue?: LiveMessagingSourceValue | null
  edit: (patch: Partial<LiveMessagingSourceValue>) => void
  save: (patch?: Partial<LiveMessagingSourceValue>) => Promise<void>
  verify: () => Promise<MessagingVerification>
  listChannels: () => Promise<MessagingChannelChoice[]>
  sync: () => Promise<string>
  remove: () => Promise<void>
}) {
  const tx = useAuthoredText()
  const service = brand === 'slack' ? 'Slack' : 'Discord'
  const persistence = useMessagingDraftState(value.enabled, save, persistedValue !== undefined ? { draftKey: liveMessagingDraftKey(value), persistedKey: persistedValue ? liveMessagingDraftKey(persistedValue) : null, enabled: persistedValue?.enabled ?? false } : undefined)
  const [state, setState] = useState<MessagingActionState>({ phase: 'idle', verified: false, message: '' })
  const controller = useRef<ReturnType<typeof createMessagingConnectionState> | null>(null)
  if (!controller.current) controller.current = createMessagingConnectionState(setState)
  const actions = controller.current
  const [channels, setChannels] = useState<MessagingChannelChoice[] | null>(null)
  const [filter, setFilter] = useState('')
  const credentialVersion = useRef(value.credentialVersion)
  useEffect(() => { actions.activate(); return () => actions.dispose() }, [actions])
  useEffect(() => {
    if (credentialVersion.current === value.credentialVersion) return
    credentialVersion.current = value.credentialVersion
    // Our own save publishes its protected credential stamp before verification finishes.
    if (actions.snapshot().phase !== 'verifying') { actions.invalidate(); setChannels(null) }
  }, [actions, value.credentialVersion])
  const busy = state.phase === 'verifying' || state.phase === 'loading' || state.phase === 'syncing' || state.phase === 'saving'
  const scopeError = messagingChannelScopeError(value.channels)
  const entries = messagingChannelEntries(value.channels)
  const selected = new Set(entries.map((entry) => entry.toLowerCase()))
  const change = (patch: Partial<LiveMessagingSourceValue>, tokenChanged = false) => {
    persistence.edited()
    actions.invalidate(!tokenChanged)
    if (tokenChanged) setChannels(null)
    edit({ ...patch, enabled: false })
  }
  const editDetails = (patch: Partial<LiveMessagingSourceValue>) => { persistence.edited(); edit(patch) }
  const connect = () => actions.action('saving', async () => { await persistence.save({ enabled: true }); return 'Connected; only the selected channels are enabled.' })
  const disconnect = async () => { if (await actions.action('saving', async () => { await persistence.save({ enabled: false }); return 'Disconnected' }, false)) actions.invalidate(false, 'Disconnected; the token stays saved for reconnection.') }
  const loadChannels = () => actions.action('loading', async () => { const next = await listChannels(); setChannels(next); return next.length ? `${next.length} channels available; select the ones TensorFold may use.` : 'No channels found; invite the bot and check channel permissions, then refresh.' })
  return <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
    <div className="flex items-start gap-3">
      <BrandIcon brand={brand} size={32} />
      <div className="min-w-0 flex-1">
        <Input aria-label={tx("{value0} connection label", { value0: service })} value={value.label} onChange={(event) => editDetails({ label: event.target.value })} disabled={busy} className="h-8" />
        <p className="mt-1 text-[13px] text-ink-3">{persistence.dirty ? persistence.persistedEnabled ? tx("Unsaved changes · Saved connection remains enabled") : tx("Unsaved connection changes") : value.enabled ? tx("Enabled for selected channels") : tx("Disconnected")}{value.lastSyncAt ? tx(" · Last read {value0}", { value0: new Date(value.lastSyncAt).toLocaleString(currentIntlLocale()) }) : tx(" · Not read yet")}</p>
      </div>
      <Button variant="danger" size="sm" aria-label={tx("Remove {value0} connection", { value0: value.label || service })} disabled={busy} onClick={() => actions.action('saving', async () => { await remove(); return 'Connection removed' }, false)}>{tx("Remove")}</Button>
    </div>
    <div className="mt-3 space-y-3">
      <Field label={tx("Bot token")} hint={tx("protected secret")}><Input aria-label={tx("{value0} bot token", { value0: service })} type="password" value={value.token} autoComplete="off" spellCheck={false} disabled={busy} onChange={(event) => change({ token: event.target.value }, true)} placeholder={brand === 'slack' ? tx("Bot token from your Slack app") : tx("Bot token from the Discord Developer Portal")} /></Field>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" icon="shield" disabled={busy || !value.token.trim()} onClick={() => actions.verify(() => persistence.save(), verify)}>{state.phase === 'verifying' ? tx("Verifying…") : state.verified ? tx("Verify again") : tx("Save and verify token")}</Button>
        <Button size="sm" icon="refresh" disabled={busy || !state.verified} onClick={loadChannels}>{state.phase === 'loading' ? tx("Loading channels…") : channels ? tx("Refresh channels") : tx("Choose channels")}</Button>
      </div>
      <Field label={tx("Allowed channels")} hint={tx("required")}><Input aria-label={tx("{value0} allowed channels", { value0: service })} value={value.channels} disabled={busy} onChange={(event) => change({ channels: event.target.value })} placeholder={tx("Select channels below, or enter their IDs")} spellCheck={false} /></Field>
      <p className="text-[12px] leading-relaxed text-ink-3">{tx("Empty means no access, including older connections with an empty list; enter * alone only if you explicitly want every channel the bot can access, including future channels.")}</p>
      {channels && <div className="rounded-lg border border-line bg-card-2 p-2.5">
        <Input aria-label={tx("Filter {value0} channels", { value0: service })} value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={tx("Find a channel")} className="mb-2 h-8" />
        <div className="max-h-52 space-y-1 overflow-y-auto">
          {channels.filter((channel) => `${channel.name} ${channel.detail ?? ''} ${channel.id}`.toLowerCase().includes(filter.toLowerCase())).map((channel) => <label key={channel.id} className="flex cursor-pointer items-start gap-2 rounded-md px-1 py-1.5 text-[13px] text-ink-2">
            <input type="checkbox" disabled={busy} checked={selected.has('*') || selected.has(channel.id.toLowerCase()) || selected.has(channel.name.toLowerCase())} onChange={(event) => {
              const next = selected.has('*') ? channels.map((item) => item.id).filter((id) => id !== channel.id) : entries.filter((entry) => entry.toLowerCase() !== channel.id.toLowerCase() && entry.toLowerCase() !== channel.name.toLowerCase())
              if (event.target.checked && !selected.has('*')) next.push(channel.id)
              change({ channels: next.join(', ') })
            }} className="mt-0.5 accent-accent" />
            <span className="min-w-0 break-words">#{channel.name}<span className="block text-[12px] text-ink-3">{channel.detail ? `${channel.detail} · ` : ''}{channel.id}</span></span>
          </label>)}
          {!channels.length && <p className="p-1 text-[13px] text-ink-3">{tx("Invite the bot to a channel and check its permissions before refreshing.")}</p>}
          {!!channels.length && !channels.some((channel) => `${channel.name} ${channel.detail ?? ''} ${channel.id}`.toLowerCase().includes(filter.toLowerCase())) && <p className="p-1 text-[13px] text-ink-3">{tx("No channels match this filter.")}</p>}
        </div>
      </div>}
      <details className="text-[13px] text-ink-3"><summary className="cursor-pointer">{tx("Optional channel guidance")}</summary><Input aria-label={tx("{value0} channel guidance", { value0: service })} className="mt-2" value={value.channelPrompt ?? ''} disabled={busy} onChange={(event) => editDetails({ channelPrompt: event.target.value })} placeholder={tx("Guidance used only for this connection")} /></details>
      {scopeError && <p className="text-[12px] text-warn">{scopeError}</p>}
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" size="sm" disabled={busy || !state.verified || !!scopeError} onClick={connect}>{value.enabled ? tx("Save connection") : value.lastSyncAt ? tx("Reconnect") : tx("Connect selected channels")}</Button>
        <Button size="sm" icon="message" disabled={busy || !state.verified || !value.enabled || !!scopeError} onClick={() => actions.action('syncing', async () => { await persistence.save(); return sync() })}>{state.phase === 'syncing' ? tx("Reading…") : tx("Read selected channels")}</Button>
        <Button size="sm" disabled={busy || (!value.enabled && !persistence.persistedEnabled)} onClick={disconnect}>{tx("Disconnect")}</Button>
      </div>
      {persistence.dirty && <p role="status" className="text-[12px] text-warn">{tx("Save and verify these changes before connecting; previous saved settings remain in effect until a save succeeds, and Disconnect remains available for an enabled saved connection.")}</p>}
      <p className="text-[12px] text-ink-3">{tx("Reads happen when you request a sync; this is a bot connection, not a personal account sign-in, and replies use the assistant approval flow.")}</p>
      {state.message && <p role={state.phase === 'error' ? 'alert' : 'status'} className={`break-words text-[13px] ${state.phase === 'error' ? 'text-danger' : 'text-ink-2'}`}>{state.message}</p>}
      {value.lastError && state.phase !== 'verifying' && <p role="alert" className="text-[13px] text-danger">{tx("The last read failed; verify the connection and check channel permissions before retrying.")}</p>}
    </div>
  </div>
}
