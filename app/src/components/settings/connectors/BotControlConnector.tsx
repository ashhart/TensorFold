import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { api } from '../../../lib/ipc'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Toggle } from '../../ui'
import { AdvancedSection, Section, type SettingsSectionContext } from '../shared'

export function BotControlConnector({ ctx, focus }: { ctx: SettingsSectionContext; focus?: 'telegram' | 'whatsapp-cloud' | 'imessage' }) {
  const tx = useAuthoredText()
  const {
    draft,
    setDraft,
    commit,
    botBusy,
    botStatus,
    commitTelegram,
    commitWhatsapp,
    pollTelegram,
    imessageBusy,
    imessageAccess,
    checkImessageAccess,
    toggleImessage,
    syncImessage,
  } = ctx

  return (
    <Section title={focus === 'imessage' ? tx("iMessage access") : focus === 'telegram' ? tx("Telegram setup") : focus === 'whatsapp-cloud' ? tx("Business Cloud setup") : tx("Bot control")} desc={tx("Choose exactly which conversations and senders TensorFold may use.")}>
      <div className="space-y-3">
        {(!focus || focus === 'telegram') && (
        <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2.5">
              <BrandIcon brand="telegram" size={32} />
              <div className="min-w-0">
                <div className="text-[13px] font-medium text-ink">{tx("Telegram")}</div>
                <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                  {draft.botControl.telegram.lastPollAt ? tx("Last poll {value0}", { value0: new Date(draft.botControl.telegram.lastPollAt).toLocaleString(currentIntlLocale()) }) : tx("Polling is local; no public webhook required.")}
                </div>
              </div>
            </div>
            <Toggle label={tx("Telegram bot control")} checked={draft.botControl.telegram.enabled} onChange={(enabled) => commitTelegram({ ...draft.botControl.telegram, enabled })} />
          </div>
          <div className="mb-3 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
            {tx("Every command must match both lists. In group chats, the group chat ID does not authorize every member: add each trusted sender's numeric Telegram user ID explicitly.")}</div>
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <Field label={tx("Bot token")}>
              <Input
                type="password"
                value={draft.botControl.telegram.botToken}
                onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, telegram: { ...d.botControl.telegram, botToken: e.target.value } } }))}
                onBlur={() => commit(draft)}
                placeholder={tx("123456:ABC...")}
                spellCheck={false}
              />
            </Field>
            <Field label={tx("Allowed chat IDs")} hint={tx("required")}>
              <Input
                value={draft.botControl.telegram.allowedChatIds}
                onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, telegram: { ...d.botControl.telegram, allowedChatIds: e.target.value } } }))}
                onBlur={() => commit(draft)}
                placeholder="123456789, -100..."
                spellCheck={false}
              />
            </Field>
            <Field label={tx("Allowed user IDs")} hint={tx("required")}>
              <Input
                value={draft.botControl.telegram.allowedUserIds}
                onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, telegram: { ...d.botControl.telegram, allowedUserIds: e.target.value } } }))}
                onBlur={() => commit(draft)}
                placeholder="123456789, 987654321"
                spellCheck={false}
              />
            </Field>
            <Field label={tx("Background polling")} hint={draft.botControl.telegram.polling ? tx("On") : tx("Off")}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">{tx("Poll Telegram every few seconds and route allowlisted commands into rooms.")}</span>
                <Toggle label={tx("Telegram background polling")} checked={draft.botControl.telegram.polling} onChange={(polling) => commitTelegram({ ...draft.botControl.telegram, polling })} />
              </div>
            </Field>
            <Field label={tx("Offset")} hint={draft.botControl.telegram.lastUpdateId == null ? tx("not started") : String(draft.botControl.telegram.lastUpdateId)}>
              <Button variant="default" icon="message" onClick={pollTelegram} disabled={botBusy === 'telegram' || !draft.botControl.telegram.enabled}>
                {botBusy === 'telegram' ? tx("Polling...") : tx("Poll now")}
              </Button>
            </Field>
          </div>
          {(draft.botControl.telegram.lastError || botStatus) && (
            <div className={clsx('mt-2.5 rounded-lg border px-3 py-2 text-[13px]', draft.botControl.telegram.lastError && !botStatus ? 'border-danger/30 bg-danger/5 text-danger' : 'border-line bg-card-2 text-ink-3')}>
              {botStatus || draft.botControl.telegram.lastError}
            </div>
          )}
        </div>
        )}

        {(!focus || focus === 'whatsapp-cloud') && (
        <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2.5">
              <BrandIcon brand="whatsapp" size={32} />
              <div className="min-w-0">
                <div className="text-[13px] font-medium text-ink">{tx("WhatsApp")}</div>
                <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{tx("Bridge-ready: route verified WhatsApp messages through the same room dispatcher.")}</div>
              </div>
            </div>
            <Toggle label={tx("WhatsApp bot control")} checked={draft.botControl.whatsapp.enabled} onChange={(enabled) => commitWhatsapp({ ...draft.botControl.whatsapp, enabled })} />
          </div>
          <AdvancedSection title={tx("Meta Cloud API credentials")} icon="shield">
            <div className="rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
              {tx("For the official Meta Business WhatsApp API. Point a TLS reverse proxy at the loopback listener route ")}<code>/integrations/whatsapp</code>{tx("; TensorFold verifies the challenge token, raw-body Meta signature, phone-number ID, and sender allowlist. For a personal phone account, use Link your phone above.")}</div>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <Field label={tx("Access token")}>
                <Input
                  type="password"
                  value={draft.botControl.whatsapp.accessToken}
                  onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, whatsapp: { ...d.botControl.whatsapp, accessToken: e.target.value } } }))}
                  onBlur={() => commit(draft)}
                  placeholder={tx("Meta token")}
                  spellCheck={false}
                />
              </Field>
              <Field label={tx("Phone number ID")}>
                <Input
                  value={draft.botControl.whatsapp.phoneNumberId}
                  onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, whatsapp: { ...d.botControl.whatsapp, phoneNumberId: e.target.value } } }))}
                  onBlur={() => commit(draft)}
                  placeholder="1234567890"
                  spellCheck={false}
                />
              </Field>
              <Field label={tx("Verify token")}>
                <Input
                  type="password"
                  value={draft.botControl.whatsapp.verifyToken}
                  onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, whatsapp: { ...d.botControl.whatsapp, verifyToken: e.target.value } } }))}
                  onBlur={() => commit(draft)}
                  placeholder={tx("webhook verification secret")}
                  spellCheck={false}
                />
              </Field>
              <Field label={tx("Meta app secret")} hint={tx("signature key")}>
                <Input
                  type="password"
                  value={draft.botControl.whatsapp.appSecret}
                  onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, whatsapp: { ...d.botControl.whatsapp, appSecret: e.target.value } } }))}
                  onBlur={() => commit(draft)}
                  placeholder={tx("Meta app secret")}
                  spellCheck={false}
                />
              </Field>
              <Field label={tx("Allowed sender IDs")}>
                <Input
                  value={draft.botControl.whatsapp.allowedChatIds}
                  onChange={(e) => setDraft((d) => ({ ...d, botControl: { ...d.botControl, whatsapp: { ...d.botControl.whatsapp, allowedChatIds: e.target.value } } }))}
                  onBlur={() => commit(draft)}
                  placeholder="447..."
                  spellCheck={false}
                />
              </Field>
            </div>
          </AdvancedSection>
        </div>
        )}

        {(!focus || focus === 'imessage') && (
        <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2.5">
              <BrandIcon brand="imessage" size={32} />
              <div className="min-w-0">
                <div className="text-[13px] font-medium text-ink">{"iMessage"}</div>
                <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                  {tx("Read your own iMessage database (~/Library/Messages/chat.db) read-only. Requires Full Disk Access.")}</div>
              </div>
            </div>
            <Toggle label={tx("iMessage connector")} checked={draft.imessage.enabled} onChange={toggleImessage} />
          </div>
          {draft.imessage.enabled && (
            <div className="mt-3 space-y-2.5">
              {/* Full Disk Access gate: probe chat.db and report the precise state. */}
              <div className={clsx('rounded-lg border px-3 py-2 text-[13px]', imessageAccess?.state === 'ok' ? 'border-positive/30 bg-positive/10 text-positive' : imessageAccess?.state === 'needs_full_disk_access' ? 'border-warn/30 bg-warn/10 text-warn' : 'border-line bg-card-2 text-ink-3')}>
                {imessageAccess ? imessageAccess.detail : tx("Click \"Check access\" to verify TensorFold can read your iMessage database.")}
              </div>
              {imessageAccess?.state === 'needs_full_disk_access' && (
                <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-[13px] leading-snug text-warn">
                  <div>{tx("Grant TensorFold Full Disk Access, then restart the app.")}</div>
                  <button onClick={() => { void api.openFullDiskAccess().catch(() => {}) }} className="mt-1.5 inline-flex items-center gap-1 rounded-lg border border-warn/40 bg-card px-2.5 py-1 text-[13px] font-medium text-warn transition-colors hover:bg-warn/10">
                    <Icon name="shield" size={12} /> {tx("Open System Settings")}</button>
                </div>
              )}
              <Field label={tx("Allowed contacts")} hint={tx("optional")}>
                <Input value={draft.imessage.allowedChatIds} onChange={(e) => setDraft((d) => ({ ...d, imessage: { ...d.imessage, allowedChatIds: e.target.value } }))} onBlur={() => commit(draft)} placeholder={tx("phone or email handle (blank = all)")} className="h-8" spellCheck={false} />
              </Field>
              <div className="flex flex-wrap gap-2">
                <Button variant="default" size="sm" icon="shield" onClick={checkImessageAccess} disabled={imessageBusy}>{imessageBusy ? tx("Checking") : tx("Check access")}</Button>
                <Button variant="primary" size="sm" icon="message" onClick={syncImessage} disabled={imessageBusy || imessageAccess?.state !== 'ok'}>{imessageBusy ? tx("Syncing") : tx("Sync iMessage")}</Button>
              </div>
              {draft.imessage.lastError && <div className="text-[13px] text-danger">{<OriginalDiagnostic>{draft.imessage.lastError}</OriginalDiagnostic>}</div>}
            </div>
          )}
        </div>
        )}
      </div>
    </Section>
  )
}
