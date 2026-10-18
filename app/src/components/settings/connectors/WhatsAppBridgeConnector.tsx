import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { useEffect, useState } from 'react'
import type { WhatsAppBridgeSource } from '../../../lib/types'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { QrCode } from '../QrCode'

function WhatsAppContactScope({ source, busy, onUpdate }: {
  source: WhatsAppBridgeSource
  busy: boolean
  onUpdate: (id: string, patch: Partial<Pick<WhatsAppBridgeSource, 'allowedChatJids' | 'allowSelfChat'>>) => unknown
}) {
  const tx = useAuthoredText()
  const serialized = (source.allowedChatJids ?? []).join('\n')
  const [contacts, setContacts] = useState(serialized)
  useEffect(() => { setContacts(serialized) }, [serialized, source.id])
  return <div className="mt-3 space-y-2 border-t border-line pt-3">
    <div className="flex items-center justify-between gap-3">
      <div><div className="text-[13px] font-medium text-ink">{tx("My own WhatsApp chat")}</div><p className="text-[12px] text-ink-3">{tx("Receive notes you send yourself from your linked phone; TensorFold replies are ignored.")}</p></div>
      <Toggle label={tx("Receive my own WhatsApp chat")} checked={source.allowSelfChat === true} onChange={(checked) => onUpdate(source.id, { allowSelfChat: checked })} disabled={busy} />
    </div>
    <label className="block text-[13px] text-ink-2" htmlFor={`wa-contacts-${source.id}`}>{tx("Selected chats")}</label>
    <textarea id={`wa-contacts-${source.id}`} value={contacts} onChange={(event) => setContacts(event.target.value)} disabled={busy} rows={3} placeholder={tx("International phone numbers, one per line")} className="w-full resize-y rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink" />
    <p className="text-[12px] text-ink-3">{tx("Only these exact chats are received, including attachments and recent history; empty receives none, and groups require their exact WhatsApp group ID.")}</p>
    <Button size="sm" disabled={busy || contacts === serialized} onClick={() => onUpdate(source.id, { allowedChatJids: contacts.split(/[\n,]/).map((value) => value.trim()).filter(Boolean) })}>{tx("Save chats")}</Button>
    <p className="text-[12px] text-ink-3">{tx("Phone messages and history are inbox content; they never grant permissions or run bot commands automatically.")}</p>
  </div>
}

export function WhatsAppBridgeConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft, whatsappBusy, whatsappQrs, whatsappPairing, whatsappRuntime, whatsappNow, whatsappLocalOnly, addWhatsAppAccount, removeWhatsAppAccount, pairWhatsApp, updateWhatsAppAccount, disconnectWhatsAppAccount } = ctx
  if (whatsappLocalOnly) return <Section title={tx("WhatsApp · Manage on your Mac")} desc={tx("Linked-device pairing and contact access are managed locally in TensorFold on the host Mac.")}><div className="rounded-xl border border-line bg-card p-4 text-[13px] leading-relaxed text-ink-2">{tx("Open TensorFold on that Mac to link or reconnect a phone, choose contacts, disconnect, or remove an account; QR codes and local pairing controls are unavailable in this remote view.")}</div></Section>
  return <Section title={tx("WhatsApp · Link my phone")} desc={tx("Connect with WhatsApp → Linked Devices, without an API key; this experimental local bridge uses the unofficial multidevice protocol.")}>
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
        <div><div className="text-[13px] font-medium text-ink">{tx("Accounts: {count}", { count: draft.whatsapp.sources.length })}</div><div className="mt-0.5 text-[13px] text-ink-3">{draft.whatsapp.sources.filter((source) => source.enabled && whatsappRuntime?.[source.id]?.phase === 'connected').length} {tx("connected now")}</div></div>
        <Button size="sm" icon="plus" onClick={addWhatsAppAccount} disabled={!!whatsappBusy}>{whatsappBusy === 'add' ? tx("Adding") : tx("Add account")}</Button>
      </div>
      {draft.whatsapp.sources.length === 0 && <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">{tx("Add an account, link your phone, then select your own chat or the contacts TensorFold may receive.")}</div>}
      {draft.whatsapp.sources.map((source) => {
        const runtime = whatsappRuntime?.[source.id]
        const qr = whatsappQrs[source.id]
        const connecting = runtime?.phase === 'connecting' || runtime?.phase === 'reconnecting' || whatsappPairing?.[source.id]
        const connected = source.enabled && runtime?.phase === 'connected'
        const active = connecting || connected || runtime?.phase === 'pairing'
        const stateLabel = connected ? tx("Connected") : runtime?.phase === 'reconnecting' ? tx("Reconnecting · attempt {value0} of 5", { value0: String(runtime.reconnectAttempt) }) : connecting ? tx("Connecting…") : runtime?.phase === 'pairing' ? (qr ? tx("Waiting for phone") : tx("QR expired · refresh to link")) : source.paired ? tx("Disconnected · phone link saved") : tx("Phone not linked")
        const error = runtime?.lastError || source.lastError
        return <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
          <div className="flex items-start gap-3">
            <BrandIcon brand="whatsapp" size={32} />
            <div className="min-w-0 flex-1">
              <Input aria-label={tx("WhatsApp account label")} value={source.label} onChange={(event) => setDraft((current) => ({ ...current, whatsapp: { ...current.whatsapp, sources: current.whatsapp.sources.map((candidate) => candidate.id === source.id ? { ...candidate, label: event.target.value } : candidate) } }))} onBlur={() => updateWhatsAppAccount(source.id, { label: source.label })} className="h-8" />
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[13px] text-ink-3" role="status"><span className={clsx('h-1.5 w-1.5 rounded-full', connected ? 'bg-positive' : connecting ? 'bg-warn animate-pulse' : 'bg-ink-4')} />{stateLabel}{(runtime?.phoneNumber || source.phoneNumber) && <span>· {runtime?.phoneNumber || source.phoneNumber}</span>}</div>
              {error && <p className="mt-1 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
            </div>
            <button disabled={!!whatsappBusy} onClick={() => removeWhatsAppAccount(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger disabled:opacity-40" aria-label={tx("Remove WhatsApp account")}><Icon name="trash" size={15} /></button>
          </div>
          {qr && <div className="mt-3 flex flex-col items-center gap-2 rounded-lg border border-line bg-card-2 p-4"><div className="rounded-lg bg-white p-3"><QrCode value={qr} size={180} /></div><p className="text-center text-[13px] text-ink-3">{tx("WhatsApp → Settings → Linked Devices → Link a Device, then scan.")}</p><p className="text-[12px] text-ink-3">{tx("Expires in ")}{Math.max(0, Math.ceil(((runtime?.qrExpiresAt ?? 0) - whatsappNow) / 1000))} {tx("seconds; a fresh code replaces it automatically.")}</p></div>}
          {connecting && <div className="mt-3 flex items-center gap-2 text-[13px] text-ink-3"><Icon name="refresh" size={13} className="animate-spin" />{runtime?.phase === 'reconnecting' ? tx("Restoring the saved phone connection…") : tx("Opening the phone connection…")}</div>}
          <div className="mt-3 flex flex-wrap gap-2">
            {!connected && !connecting && <Button variant="primary" size="sm" onClick={() => pairWhatsApp(source.id)} disabled={!!whatsappBusy}>{source.paired ? tx("Reconnect") : runtime?.phase === 'pairing' ? tx("Refresh QR") : tx("Link phone")}</Button>}
            {active && <Button size="sm" onClick={() => disconnectWhatsAppAccount(source.id)} disabled={!!whatsappBusy}>{tx("Disconnect")}</Button>}
          </div>
          <WhatsAppContactScope source={source} busy={!!whatsappBusy} onUpdate={updateWhatsAppAccount} />
        </div>
      })}
    </div>
  </Section>
}
