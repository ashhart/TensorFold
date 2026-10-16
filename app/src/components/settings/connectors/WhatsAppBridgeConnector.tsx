import clsx from 'clsx'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Input } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { QrCode } from '../QrCode'

export function WhatsAppBridgeConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const {
    draft,
    setDraft,
    commit,
    whatsappBusy,
    whatsappQrs,
    whatsappPairing,
    addWhatsAppAccount,
    removeWhatsAppAccount,
    pairWhatsApp,
  } = ctx

  return (
    <Section title="WhatsApp bridge (multi-account)" desc="Pair one or more WhatsApp numbers by scanning a QR code (WhatsApp → Linked Devices). Experimental — uses the multidevice protocol; bridge your own accounts at your own discretion.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.whatsapp.sources.length} account{draft.whatsapp.sources.length === 1 ? '' : 's'}</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.whatsapp.sources.filter((s) => s.paired).length} paired</div>
          </div>
          <Button variant="default" size="sm" icon="plus" onClick={addWhatsAppAccount} disabled={!!whatsappBusy}>{whatsappBusy === 'add' ? 'Adding' : 'Add account'}</Button>
        </div>

        {draft.whatsapp.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add an account, then scan the QR with your phone to pair. Each account is isolated with its own credentials.
          </div>
        ) : draft.whatsapp.sources.map((source) => {
          const qr = whatsappQrs[source.id]
          // Bridge is starting but hasn't yet emitted its first QR/status. Keeps the card in a
          // visible "Connecting" state instead of flicking back to idle (the startPairing IPC
          // returns immediately while the QR is still being fetched a moment later).
          const connecting = Boolean(whatsappPairing?.[source.id]) && !qr && !source.paired
          return (
            <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="flex items-start gap-3">
                <BrandIcon brand="whatsapp" size={32} />
                <div className="min-w-0 flex-1">
                  <Input
                    value={source.label}
                    onChange={(e) => setDraft((d) => ({ ...d, whatsapp: { ...d.whatsapp, sources: d.whatsapp.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                    onBlur={() => commit(draft)}
                    className="h-8"
                  />
                  <div className="mt-1 flex items-center gap-1.5 text-[13px] text-ink-3">
                    <span className={clsx('h-1.5 w-1.5 rounded-full', source.paired ? 'bg-positive' : connecting ? 'bg-warn animate-pulse' : 'bg-ink-4')} />
                    {source.paired ? `Paired${source.phoneNumber ? ' · ' + source.phoneNumber : ''}` : connecting ? 'Connecting…' : 'Not paired'}
                    {source.lastError && <span className="text-danger">· {source.lastError}</span>}
                  </div>
                </div>
                <button onClick={() => removeWhatsAppAccount(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove WhatsApp account">
                  <Icon name="trash" size={15} />
                </button>
              </div>
              {/* QR pairing: scan with WhatsApp → Linked Devices → Link a Device. */}
              {qr ? (
                <div className="mt-3 flex flex-col items-center gap-2 rounded-lg border border-line bg-card-2 p-4">
                  <div className="rounded-lg bg-white p-3">
                    {/* The QR string is rendered by the renderer via an inline data URI so
                        no third-party QR library is bundled. Falls back to raw text below. */}
                    <QrCode value={qr} size={180} />
                  </div>
                  <div className="text-center text-[13px] leading-snug text-ink-3">Open WhatsApp on your phone → Settings → Linked Devices → Link a Device, then scan this code.</div>
                </div>
              ) : connecting ? (
                // Waiting for the bridge to produce the first QR. Without this the user sees
                // nothing between the click and the code appearing — the original "does nothing".
                <div className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-3">
                  <Icon name="refresh" size={13} className="animate-spin" />
                  Connecting to WhatsApp — waiting for the QR code…
                </div>
              ) : (
                <div className="mt-3 flex gap-2">
                  <Button variant="primary" size="sm" icon="plus" onClick={() => pairWhatsApp(source.id)} disabled={whatsappBusy === 'pair-' + source.id || source.paired}>
                    {whatsappBusy === 'pair-' + source.id ? 'Starting' : source.paired ? 'Paired' : source.lastError ? 'Retry pairing' : 'Start pairing'}
                  </Button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Section>
  )
}
