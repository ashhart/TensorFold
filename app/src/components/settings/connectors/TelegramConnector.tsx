import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../../lib/ipc'
import { BrandIcon } from '../../../lib/brand'
import { Button, Field, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import type { Settings } from '../../../lib/types'
import { MessagingSetupGuide } from './MessagingSetupGuide'
import { createMessagingConnectionState, telegramAccessError, type MessagingActionState } from './messaging-connection-state'
import { TelegramRemoteSettingsPanel } from './TelegramRemoteSettingsPanel'
import { telegramMessagingDraftKey, useMessagingDraftState } from './useMessagingDraftState'

type TelegramSettings = Settings['botControl']['telegram']
export function TelegramConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const telegram = ctx.draft.botControl.telegram
  const [state, setState] = useState<MessagingActionState>({ phase: 'idle', verified: false, message: '' })
  const controller = useRef<ReturnType<typeof createMessagingConnectionState> | null>(null)
  if (!controller.current) controller.current = createMessagingConnectionState(setState)
  const actions = controller.current
  const credentialVersion = useRef(telegram.secretRotations?.botToken)
  useEffect(() => { actions.activate(); return () => actions.dispose() }, [actions])
  useEffect(() => {
    if (credentialVersion.current === telegram.secretRotations?.botToken) return
    credentialVersion.current = telegram.secretRotations?.botToken
    if (actions.snapshot().phase !== 'verifying') actions.invalidate()
  }, [actions, telegram.secretRotations?.botToken])
  const busy = ['verifying', 'saving', 'syncing', 'loading'].includes(state.phase)
  const accessError = telegramAccessError(telegram.allowedChatIds, telegram.allowedUserIds)
  const saveDraft = async (patch: Partial<TelegramSettings> = {}) => {
    await ctx.save({ ...ctx.draft, botControl: { ...ctx.draft.botControl, telegram: { ...telegram, ...patch } } })
  }
  const persistedTelegram = ctx.persistedSettings?.botControl.telegram
  const persistence = useMessagingDraftState(telegram.enabled, saveDraft, persistedTelegram ? { draftKey: telegramMessagingDraftKey(telegram), persistedKey: telegramMessagingDraftKey(persistedTelegram), enabled: persistedTelegram.enabled } : undefined)
  const save = persistence.save
  const edit = (patch: Partial<TelegramSettings>, tokenChanged = false) => {
    persistence.edited()
    actions.invalidate(!tokenChanged)
    ctx.setDraft((current) => ({ ...current, botControl: { ...current.botControl, telegram: { ...current.botControl.telegram, ...patch, enabled: false, polling: false } } }))
  }
  const disconnect = async (forgetToken = false) => {
    if (await actions.action('saving', async () => {
      await save({ enabled: false, polling: false, ...(forgetToken ? { botToken: '', lastUpdateId: null, lastPollAt: null, lastError: null } : {}) })
      return 'Disconnected'
    }, false)) actions.invalidate(false, forgetToken ? 'Disconnected and the saved bot token was removed from TensorFold; revoke it in BotFather if needed.' : 'Disconnected; the bot token and allowlists stay saved for reconnection.')
  }
  return <Section title={tx("Telegram")} desc={tx("Connect your own Telegram bot, then choose the chats and people allowed to control TensorFold.")}>
    <div className="space-y-3">
      <MessagingSetupGuide steps={[
        'Open the verified @BotFather account in Telegram, create a bot with /newbot and copy the token; this connects a bot, not your personal Telegram account.',
        'Start a private chat with your bot or add it to a group; keep group privacy enabled unless you explicitly need all group messages.',
        'Save and verify the token, then enter the exact numeric chat IDs and sender IDs you trust; both lists must match before a command is accepted.',
      ]} links={[{ label: tx("Open BotFather"), url: 'https://t.me/BotFather' }, { label: tx("Telegram bot setup"), url: 'https://core.telegram.org/bots/tutorial' }]} />
      <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
        <div className="flex items-start gap-3"><BrandIcon brand="telegram" size={32} /><div className="min-w-0"><div className="text-[13px] font-medium text-ink">{tx("Telegram bot")}</div><p className="mt-0.5 text-[13px] text-ink-3">{persistence.dirty ? persistence.persistedEnabled ? tx("Unsaved changes · Saved bot connection remains enabled") : tx("Unsaved bot connection changes") : telegram.enabled ? telegram.polling ? tx("Connected · Background checks enabled") : tx("Connected · Manual checks") : tx("Disconnected")}{telegram.lastPollAt ? tx(" · Last checked {value0}", { value0: new Date(telegram.lastPollAt).toLocaleString(currentIntlLocale()) }) : ''}</p></div></div>
        <div className="mt-3 space-y-3">
          <Field label={tx("Bot token")} hint={tx("protected secret")}><Input aria-label={tx("Telegram bot token")} type="password" value={telegram.botToken} autoComplete="off" spellCheck={false} disabled={busy} onChange={(event) => edit({ botToken: event.target.value, lastUpdateId: null, lastPollAt: null }, true)} placeholder={tx("Token provided by BotFather")} /></Field>
          <Button size="sm" icon="shield" disabled={busy || !telegram.botToken.trim()} onClick={() => actions.verify(() => save(), async () => {
            const result = await api.verifyTelegram()
            return { ok: result.ok, identity: result.username ? `@${result.username}` : undefined, error: result.ok ? undefined : 'Telegram could not verify this bot; check the token, remove any conflicting webhook and retry.' }
          })}>{state.phase === 'verifying' ? tx("Verifying…") : state.verified ? tx("Verify again") : tx("Save and verify token")}</Button>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={tx("Allowed chat IDs")} hint={tx("required")}><Input aria-label={tx("Telegram allowed chat IDs")} value={telegram.allowedChatIds} disabled={busy} onChange={(event) => edit({ allowedChatIds: event.target.value })} placeholder={tx("Private or group chat numeric IDs")} spellCheck={false} /></Field>
            <Field label={tx("Allowed sender IDs")} hint={tx("required")}><Input aria-label={tx("Telegram allowed sender IDs")} value={telegram.allowedUserIds} disabled={busy} onChange={(event) => edit({ allowedUserIds: event.target.value })} placeholder={tx("Trusted people’s numeric user IDs")} spellCheck={false} /></Field>
          </div>
          <p className="text-[12px] leading-relaxed text-ink-3">{tx("Every command must match both lists; adding a group does not authorize every member, and empty lists authorize nobody.")}</p>
          <p className="text-[12px] leading-relaxed text-ink-3">{tx("Telegram has no chat picker for a new bot; obtain IDs from your own bot updates using Telegram’s Bot API documentation, without sharing the bot token or sending messages to third-party ID bots.")}</p>
          {accessError && <p className="text-[12px] text-warn">{accessError}</p>}
          <div className="flex items-center justify-between gap-3 rounded-lg border border-line bg-card-2 p-3"><div><p className="text-[13px] text-ink-2">{tx("Check for new commands automatically")}</p><p className="mt-0.5 text-[12px] text-ink-3">{tx("Local polling; no public webhook is required.")}</p></div><Toggle label={tx("Telegram background polling")} checked={telegram.polling} disabled={busy || !state.verified || !!accessError} onChange={(polling) => actions.action('saving', async () => { await save({ polling }); return polling ? 'Background checks enabled' : 'Background checks paused; you can still check manually.' })} /></div>
          <div className="flex flex-wrap gap-2"><Button size="sm" variant="primary" disabled={busy || !state.verified || !!accessError} onClick={() => actions.action('saving', async () => { await save({ enabled: true }); return 'Telegram connected with the selected chat and sender access.' })}>{telegram.enabled ? tx("Save connection") : telegram.lastPollAt ? tx("Reconnect bot") : tx("Connect bot")}</Button><Button size="sm" icon="refresh" disabled={busy || !state.verified || !telegram.enabled || !!accessError} onClick={() => actions.action('syncing', async () => {
            await save()
            const result = await api.pollTelegram()
            if (!result.ok) throw new Error(tx("Telegram check failed"))
            return `Processed ${result.processed} updates; skipped ${result.skipped} updates that did not match access or command rules.`
          })}>{state.phase === 'syncing' ? tx("Checking…") : tx("Check now")}</Button><Button size="sm" disabled={busy || (!telegram.enabled && !persistence.persistedEnabled)} onClick={() => disconnect()}>{tx("Disconnect")}</Button><Button size="sm" variant="danger" disabled={busy || (!telegram.botToken && !persistence.dirty && !persistence.persistedEnabled)} onClick={() => disconnect(true)}>{tx("Remove token")}</Button></div>
          {persistence.dirty && <p role="status" className="text-[12px] text-warn">{tx("These are unsaved edits; the saved bot settings and any enabled polling remain active until a save succeeds, so use Disconnect to stop the saved connection.")}</p>}
          {state.message && <p role={state.phase === 'error' ? 'alert' : 'status'} className={`text-[13px] ${state.phase === 'error' ? 'text-danger' : 'text-ink-2'}`}>{state.message}</p>}
          {telegram.lastError && <p role="alert" className="text-[13px] text-danger">{tx("The last Telegram check failed; verify the token, access lists and network before retrying.")}</p>}
        </div>
      </div>
      <TelegramRemoteSettingsPanel ctx={ctx} connected={telegram.enabled && state.verified && !accessError} />
      <p className="text-[12px] leading-relaxed text-ink-3">{tx("Telegram bot chats are not end-to-end encrypted; keep your TensorFold Mac awake and running for remote commands, and use only trusted chats and senders.")}</p>
    </div>
  </Section>
}
