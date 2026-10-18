import { statusText } from '../../../lib/preferences/display'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../../lib/ipc'
import { Button, Toggle } from '../../ui'
import type { SettingsSectionContext } from '../shared'
import type { TelegramRemotePublicStatus } from '../../../../shared/telegram-remote'

export function TelegramRemoteSettingsPanel({ ctx: _ctx, connected }: { ctx: SettingsSectionContext; connected: boolean }) {
  const tx = useAuthoredText()
  const [remote, setRemote] = useState<TelegramRemotePublicStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [dirty, setDirty] = useState(false)
  useEffect(() => {
    let current = true
    setBusy(true)
    void api.listTelegramRemoteTargets().then((result) => { if (current) { setRemote(result); setError('') } }).catch(() => { if (current) setError(tx("Could not load remote access settings; retry to see which chats and rooms can be shared.")) }).finally(() => { if (current) setBusy(false) })
    return () => { current = false }
  }, [])
  const refresh = async () => {
    setBusy(true); setError(''); setMessage('')
    try { setRemote(await api.listTelegramRemoteTargets()); setDirty(false) }
    catch { setError(tx("Could not refresh remote access settings; check the host connection and retry.")) }
    finally { setBusy(false) }
  }
  const update = (patch: Partial<Pick<TelegramRemotePublicStatus, 'enabled' | 'scope' | 'threadIds' | 'roomIds'>>) => {
    setRemote((current) => current ? { ...current, ...patch } : current)
    setDirty(true); setMessage('')
  }
  const save = async (disable = false) => {
    if (!remote) return
    setBusy(true); setError(''); setMessage('')
    try {
      const next = await api.configureTelegramRemote({ enabled: disable ? false : remote.enabled, scope: remote.scope, threadIds: remote.threadIds, roomIds: remote.roomIds })
      setRemote(next); setDirty(false)
      setMessage(next.enabled ? tx("Remote access saved; only this scope is available in your private bot chat.") : tx("Remote access is off; existing remote sessions were revoked."))
    } catch { setError(tx("Could not save remote access; the previous host settings remain in effect, so retry or turn access off.")) }
    finally { setBusy(false) }
  }
  const toggleId = (kind: 'threadIds' | 'roomIds', id: string, checked: boolean) => {
    if (!remote) return
    update({ [kind]: checked ? [...new Set([...remote[kind], id])] : remote[kind].filter((entry) => entry !== id) })
  }
  return <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
    <div className="flex items-start justify-between gap-3"><div><h3 className="text-[13px] font-medium text-ink">{tx("Chief of staff and room access")}</h3><p className="mt-0.5 text-[13px] leading-relaxed text-ink-3">{tx("Use your private Telegram bot chat to check progress and switch between approved chats and rooms.")}</p></div><Toggle label={tx("Enable Telegram remote access")} checked={remote?.enabled ?? false} disabled={busy || !remote || !connected} onChange={(enabled) => update({ enabled })} /></div>
    {!connected && <p className="mt-2 text-[12px] text-warn">{tx("Connect and verify the bot first; remote access additionally requires a private chat whose numeric chat ID matches an allowed sender ID.")}</p>}
    {busy && !remote && <p role="status" className="mt-3 text-[13px] text-ink-3">{tx("Loading available chats and rooms…")}</p>}
    {remote && <div className="mt-3 space-y-3">
      <p className="text-[12px] leading-relaxed text-ink-3">{tx("Remote access is separate from regular bot control; group chats cannot use this mode, Google-history, phone, Incognito, private and archived chats are excluded by the host, and tools keep their existing approval rules in the app.")}</p>
      <fieldset disabled={busy} className="space-y-2 text-[13px] text-ink-2"><legend className="mb-1 font-medium text-ink">{tx("What the bot can access")}</legend>
        <label className="flex items-start gap-2"><input type="radio" name="telegram-remote-scope" checked={remote.scope === 'selected'} onChange={() => update({ scope: 'selected' })} className="mt-0.5 accent-accent" /><span>{tx("Only the chats and rooms I select")}</span></label>
        <label className="flex items-start gap-2"><input type="radio" name="telegram-remote-scope" checked={remote.scope === 'all-eligible'} onChange={() => update({ scope: 'all-eligible' })} className="mt-0.5 accent-accent" /><span>{tx("All eligible chats and rooms, including ones I create later")}</span></label>
      </fieldset>
      {remote.scope === 'selected' && <div className="grid gap-3 sm:grid-cols-2">
        <fieldset disabled={busy} className="min-w-0 rounded-lg border border-line p-2.5"><legend className="px-1 text-[13px] text-ink">{tx("Chats")}</legend><div className="max-h-48 space-y-2 overflow-y-auto">{remote.eligibleThreads.map((thread) => <label key={thread.id} className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={remote.threadIds.includes(thread.id)} onChange={(event) => toggleId('threadIds', thread.id, event.target.checked)} className="mt-0.5 accent-accent" /><span className="min-w-0 break-words">{thread.title || tx("Untitled chat")}</span></label>)}{!remote.eligibleThreads.length && <p className="text-[12px] text-ink-3">{tx("No eligible chats yet.")}</p>}</div></fieldset>
        <fieldset disabled={busy} className="min-w-0 rounded-lg border border-line p-2.5"><legend className="px-1 text-[13px] text-ink">{tx("Rooms")}</legend><div className="max-h-48 space-y-2 overflow-y-auto">{remote.eligibleRooms.map((room) => <label key={room.id} className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={remote.roomIds.includes(room.id)} onChange={(event) => toggleId('roomIds', room.id, event.target.checked)} className="mt-0.5 accent-accent" /><span className="min-w-0 break-words">{room.name || tx("Untitled room")}<span className="block text-[12px] text-ink-3">{statusText(room.state, tx)}</span></span></label>)}{!remote.eligibleRooms.length && <p className="text-[12px] text-ink-3">{tx("No eligible rooms yet.")}</p>}</div></fieldset>
      </div>}
      {remote.scope === 'selected' && !remote.threadIds.length && !remote.roomIds.length && <p className="text-[12px] text-warn">{tx("No chats or rooms selected; the chief of staff will have no conversation access.")}</p>}
      <div className="rounded-lg border border-line bg-card-2 p-3 text-[12px] leading-relaxed text-ink-3"><p><code>/chief</code> {tx("returns to the chief of staff; ")}<code>/rooms</code> {tx("and ")}<code>/chats</code> {tx("list the shared targets; ")}<code>/room id</code> {tx("and ")}<code>/chat id</code> {tx("switch to one target.")}</p><p className="mt-1">{tx("Only allowlisted private senders can issue commands; message content never changes these permissions.")}</p></div>
      {dirty && <p role="status" className="text-[12px] text-warn">{tx("Unsaved remote access changes; the previous host permissions remain active until you save.")}</p>}
      <div className="flex flex-wrap gap-2"><Button size="sm" variant="primary" disabled={busy || !dirty || (remote.enabled && !connected)} onClick={() => save()}>{tx("Save remote access")}</Button><Button size="sm" disabled={busy} onClick={refresh}>{dirty ? tx("Discard changes and refresh") : tx("Refresh chats and rooms")}</Button><Button size="sm" disabled={busy || !remote.enabled} onClick={() => save(true)}>{tx("Turn remote access off")}</Button></div>
    </div>}
    {!remote && !busy && <Button className="mt-3" size="sm" onClick={refresh}>{tx("Retry loading access")}</Button>}
    {error && <p role="alert" className="mt-2 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {message && <p role="status" className="mt-2 text-[13px] text-ink-2">{message}</p>}
  </div>
}
