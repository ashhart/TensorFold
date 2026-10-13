import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { discardPendingDraft } from '../lib/draft-approval'
import type { AssistantControlSnapshot, ConnectorEvent, ConnectorId } from '../lib/types'
import { resolveAssistantModel } from '../lib/assistant-model'
import { MessageCardView, bubbleToCard } from './assistant/MessageCard'
import { eventToBubble, mergeBubbles, groupEventsIntoThreads, combineItemsByThread, type BubbleRow } from './assistant/conversation'
import { PendingApprovalsBar } from './assistant/PendingApprovalsBar'
import { itemFromAttention, itemFromEvent, mergeSourceRows, sourceFromConnector, sourceName } from './assistant/mappings'
import { assistantTaskContext, emailThreadContext, quickActionsFor } from './assistant/task-context'
import { AssistantHeader } from './assistant/AssistantHeader'
import { ImessageFdaBanner, WhatsappQrBanner } from './assistant/Banners'
import { TriageRail } from './assistant/TriageRail'
import { Composer } from './assistant/Composer'
import type { AssistantItem, SourceKey } from './assistant/types'

export function Assistant() {
  const launch = useStore((s) => s.launch)
  const newChat = useStore((s) => s.newChat)
  const settings = useStore((s) => s.settings)

  // Source filter lives in the store so the app sidebar drives it (the Assistant no
  // longer has its own sources panel — single sidebar).
  const selectedSource = useStore((s) => s.assistantSource) as SourceKey
  const [selectedItemId, setSelectedItemId] = useState('')
  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [taskText, setTaskText] = useState('')
  const [taskBusy, setTaskBusy] = useState(false)
  const [sendingReply, setSendingReply] = useState(false)
  const [resolvingEmail, setResolvingEmail] = useState(false)
  const [syncingEmail, setSyncingEmail] = useState(false)
  const [notice, setNotice] = useState('')

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      setSnapshot(await api.getAssistantSnapshot())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const sourceRows = useMemo(() => {
    const rows = snapshot ? snapshot.connectors.map(sourceFromConnector) : []
    return mergeSourceRows(rows)
  }, [snapshot])

  const attentionItems = useMemo(() => {
    if (!snapshot) return []
    const mapped = snapshot.attention?.length
      ? snapshot.attention.map(itemFromAttention)
      : groupEventsIntoThreads(snapshot.events).slice(0, 50).map(({ event, count }) => ({
          ...itemFromEvent(event),
          id: (event.threadKey && event.threadKey.trim()) || event.id,
          count,
        }))
    // Combine rows from the same conversation (threadKey) into ONE thread row with a
    // message count — not one row per message. Applies to the attention path too
    // (iMessage/Slack/etc. each emit one attention item per message). Keep the id
    // stable per conversation so selection survives new messages arriving.
    return combineItemsByThread(mapped)
  }, [snapshot])

  const filteredItems = useMemo(() => (
    selectedSource === 'all' ? attentionItems : attentionItems.filter((item) => item.source === selectedSource)
  ), [attentionItems, selectedSource])

  const selectedItem = filteredItems.find((item) => item.id === selectedItemId) ?? filteredItems[0] ?? null
  const selectedEvent = selectedItem?.eventId ? snapshot?.events.find((event) => event.id === selectedItem.eventId) ?? null : null
  // A two-way comms thread (not docs/calendar) can take a direct reply (connector:reply).
  const canDirectReply = !!selectedItem?.threadKey && (['email', 'slack', 'discord', 'whatsapp', 'imessage'] as ConnectorId[]).includes(selectedItem.source)

  useEffect(() => {
    if (filteredItems[0] && !filteredItems.some((item) => item.id === selectedItemId)) {
      setSelectedItemId(filteredItems[0].id)
    }
  }, [filteredItems, selectedItemId])

  const selectedSourceRow = selectedSource === 'all' ? null : sourceRows.find((source) => source.id === selectedSource) ?? null
  const openSignals = filteredItems.length
  const urgentCount = filteredItems.filter((item) => item.priority === 'urgent').length

  // Unified conversation bubbles for the selected thread: the source messages of
  // the selected connector event/thread (gray in) merged with the agent-task chat
  // for it (your instructions blue out, TensorFold's replies gray in). "Both together".
  const conversationBubbles: BubbleRow[] = useMemo(() => {
    const rows: BubbleRow[] = []
    if (snapshot) {
      // Source messages for the selected thread, or — when no thread is picked — the
      // latest events SCOPED TO THE SELECTED SOURCE (not all events, which would show
      // e.g. iMessages after switching to Email and finding no email thread).
      const scoped = selectedItem?.threadKey
        ? snapshot.events.filter((e) => e.threadKey === selectedItem.threadKey)
        : selectedSource === 'all'
          ? snapshot.events.slice(0, 12)
          : snapshot.events.filter((e) => e.connectorId === selectedSource).slice(0, 12)
      for (const e of scoped) rows.push(eventToBubble({
        id: e.id, actor: e.actor, title: e.title, summary: e.summary, detail: e.summary,
        action: e.action, direction: e.direction, connectorId: e.connectorId,
        threadKey: e.threadKey ?? undefined, createdAt: e.createdAt,
        approvalStatus: e.approvalStatus ?? undefined,
      }))
    }
    return mergeBubbles(rows)
  }, [snapshot, selectedItem?.threadKey, selectedSource])

  // Discord-style cards: each thread is its own context. The selected thread's
  // messages render as left-aligned avatar-led cards (TensorFold marked "APP"), with
  // collapsed reply previews. Built from the same bubble mapping, converted to cards.
  const conversationCards = useMemo(() => conversationBubbles.map(bubbleToCard), [conversationBubbles])

  const quickActions = quickActionsFor(selectedSource)

  const draftTask = (text: string) => {
    setTaskText(text)
    setNotice('')
  }

  // Direct connector reply — send the user's OWN message to the thread (iMessage,
  // WhatsApp, Slack, Discord, email) without launching an agent task. Routed by the
  // threadKey prefix in connector:reply. Confirmed before it leaves the machine.
  const sendReply = async () => {
    const text = taskText.trim()
    const key = selectedItem?.threadKey
    if (!text || !key || sendingReply) return
    if (!confirm(`Send this reply to ${selectedItem!.from} via ${sourceName(selectedItem!.source)}?`)) return
    setSendingReply(true)
    setNotice('')
    try {
      const r = await api.replyToThread(key, text)
      if (r.ok) { setTaskText(''); setNotice(`Sent to ${selectedItem!.from}.`); void refresh() }
      else setNotice(r.error || 'Could not send the reply.')
    } catch (e: any) {
      setNotice(e?.message ?? 'Could not send the reply.')
    } finally {
      setSendingReply(false)
    }
  }

  const startTask = async () => {
    const text = taskText.trim()
    if (!text) return
    setTaskBusy(true)
    setNotice('')
    try {
      if (settings?.activeProviderId && settings.activeModel) {
        const scope = selectedSource === 'all' ? 'all assistant sources' : sourceName(selectedSource)
        let sourceContext = assistantTaskContext(scope, selectedItem, selectedEvent, filteredItems, snapshot?.briefing)
        if (selectedItem?.source === 'email' && selectedItem.threadKey) {
          try {
            const thread = await api.readEmailThread(selectedItem.threadKey)
            sourceContext = `${sourceContext}\n\n${emailThreadContext(thread)}`
          } catch (err) {
            sourceContext = `${sourceContext}\n\n<email_thread_read_error>${err instanceof Error ? err.message : String(err)}</email_thread_read_error>`
          }
        }
        await launch({
          text: [
            `Assistant task for ${scope}.`,
            sourceContext,
            '',
            'User request:',
            text,
            '',
            'Work draft-first. Do not send, delete, purchase, schedule, or change external state without an explicit approval step.',
          ].join('\n'),
          cwd: settings.defaultCwd || '',
          // Use the dedicated Assistant model when set; the ?? keeps types non-null (active* are
          // already guaranteed by the guard above) and is the same value the resolver falls back to.
          providerId: resolveAssistantModel(settings).providerId ?? settings.activeProviderId,
          providerAccountId: resolveAssistantModel(settings).providerAccountId,
          model: resolveAssistantModel(settings).model ?? settings.activeModel,
          approvalMode: settings.approvalMode,
          projectId: null,
          // Originating connector conversation (slack:<src>:<chan>:<ts>, discord:<src>:<chan>,
          // email:<root>, …). Lets the host inject that source's per-channel ephemeral prompt
          // for THIS run only — never baked into the persisted system-prompt snapshot.
          sourceThreadKey: selectedItem?.threadKey,
          persistActiveModel: false,
        })
      } else {
        await newChat(null)
      }
      setTaskText('')
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err))
    } finally {
      setTaskBusy(false)
    }
  }

  const syncEmail = async () => {
    setSyncingEmail(true)
    setError('')
    setNotice('')
    try {
      const result = await api.syncEmail()
      await refresh()
      setNotice(`Email synced: ${result.messages} messages scanned, ${result.events.length} new events.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSyncingEmail(false)
    }
  }

  // Sync the active messaging connectors so fresh iMessage/WhatsApp/email messages
  // flow into the bubble conversation. No-ops for connectors that aren't enabled.
  const [syncingSources, setSyncingSources] = useState(false)
  const syncAllSources = async () => {
    setSyncingSources(true)
    setNotice('')
    const results: string[] = []
    try {
      if (settings?.email.sources.some((s) => s.enabled)) {
        try { const r = await api.syncEmail(); results.push(`${r.events.length} new email`) } catch { /* per-source best effort */ }
      }
      if (settings?.imessage.enabled) {
        // Check Full Disk Access before attempting the iMessage sync — without it the
        // read silently fails, so surface the grant-access popup instead.
        const access = await api.imessageAccess().catch(() => ({ state: 'error', detail: 'Could not check iMessage access.' }))
        if (access.state === 'needs_full_disk_access') {
          setNeedsFdaAccess(true)
          results.push('iMessage: needs Full Disk Access')
        } else {
          try { const r = await api.syncImessage(); results.push(`${r.messages} iMessage`) } catch (e: any) { results.push(`iMessage: ${e?.message ?? 'sync failed'}`) }
        }
      }
      await refresh()
      setNotice(results.length ? `Synced: ${results.join(', ')}.` : 'Nothing to sync — enable a connector in Sources.')
    } finally {
      setSyncingSources(false)
    }
  }

  // WhatsApp QR pairing surfacing inline: when the user starts pairing an account,
  // the QR arrives here so they can scan without leaving the Assistant.
  const [whatsappQrs, setWhatsappQrs] = useState<Record<string, string>>({})
  useEffect(() => {
    const off = api.onWhatsAppQr(({ sourceId, qr }) => setWhatsappQrs((q) => ({ ...q, [sourceId]: qr })))
    const offStatus = api.onWhatsAppStatus(({ sourceId, paired }) => {
      if (paired) setWhatsappQrs((q) => { const next = { ...q }; delete next[sourceId]; return next })
    })
    return () => { off(); offStatus() }
  }, [])

  // iMessage Full Disk Access gate: if chat.db can't be read, show a popup that
  // opens the macOS pane to grant it (Bug 3). The user can't read iMessage until FDA is on.
  const [needsFdaAccess, setNeedsFdaAccess] = useState(false)
  const openFdaPane = () => { void api.openFullDiskAccess().catch(() => {}) }

  const resolveSelectedEmail = async () => {
    if (!selectedItem?.threadKey) return
    setResolvingEmail(true)
    setError('')
    setNotice('')
    try {
      await api.resolveEmailThread({ id: selectedItem.threadKey, note: `Resolved from Assistant: ${selectedItem.title}` })
      await refresh()
      setNotice('Email thread marked resolved locally.')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setResolvingEmail(false)
    }
  }

  // Draft approval flow for blue (out) bubbles that are agent drafts awaiting you.
  // Approving sends the draft through the connector approval path (e.g. email outbox);
  // discarding drops it. The snapshotted event carries the id to resolve.
  const approveDraft = async (id: string) => {
    const event = snapshot?.events.find((e) => e.id === id && e.requiresApproval)
    if (!event) { setNotice('This draft has no pending approval to resolve.'); return }
    setError(''); setNotice('')
    try {
      await api.resolveConnectorApproval(event.id, 'approve')
      await refresh()
      setNotice('Draft approved.')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }
  const discardDraft = async (id: string) => {
    const event = snapshot?.events.find((e) => e.id === id && e.requiresApproval)
    if (!event) {
      setError('This draft has no pending approval to discard.')
      return
    }
    setError(''); setNotice('')
    try {
      await discardPendingDraft((eventId) => api.resolveConnectorApproval(eventId, 'deny'), refresh, event.id)
      setNotice('Draft discarded.')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }
  // "Edit" a draft: load its text into the composer so you can revise + resend.
  const draftTextFor = (id: string): string => {
    const row = conversationBubbles.find((r) => r.id === id)
    return row?.text ?? ''
  }
  const editDraft = (id: string) => {
    const text = draftTextFor(id)
    if (!text) { setNotice('Nothing to edit in this draft.'); return }
    setTaskText(text)
    setNotice('Draft loaded into the composer — revise and send.')
  }
  // Ask the assistant to redraft a pending reply in a different register.
  const redraftDraft = (id: string, how: string) => {
    const text = draftTextFor(id)
    draftTask(`Redraft this reply to be ${how}. Keep the same intent and recipient; reply with the new draft only.\n\n---\n${text}`)
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <main className="flex min-h-0 flex-1 flex-col">
        <AssistantHeader
          snapshot={snapshot}
          selectedSourceRow={selectedSourceRow}
          selectedItem={selectedItem}
          selectedSource={selectedSource}
          openSignals={openSignals}
          urgentCount={urgentCount}
          loading={loading}
          syncingSources={syncingSources}
          syncingEmail={syncingEmail}
          onSyncAll={syncAllSources}
          onRefresh={refresh}
          onSyncEmail={syncEmail}
          onBriefMe={draftTask}
        />

        {error && <div className="mx-8 mt-4 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</div>}
        {notice && <div className="mx-8 mt-4 rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2">{notice}</div>}

        {/* WhatsApp QR pairing surfaced inline: scan with your phone to pair. */}
        {Object.keys(whatsappQrs).length > 0 && (
          <WhatsappQrBanner qr={Object.values(whatsappQrs)[0]} />
        )}

        {/* iMessage Full Disk Access popup: chat.db can't be read until the user
            grants TensorFold FDA. Offer to open the pane directly (Bug 3). */}
        {needsFdaAccess && (
          <ImessageFdaBanner onOpen={() => { void openFdaPane() }} onDismiss={() => setNeedsFdaAccess(false)} />
        )}

        {/* Consolidated approve-from-the-channel queue: every blocked outbound action
            across all connectors, resolvable inline without opening each thread. */}
        <PendingApprovalsBar
          events={snapshot?.events ?? []}
          onResolved={() => { void refresh() }}
          onOpen={(threadKey) => setSelectedItemId(threadKey)}
        />

        <div className="flex min-h-0 flex-1">
          <TriageRail
            items={filteredItems}
            selectedItemId={selectedItem?.id ?? null}
            onSelect={setSelectedItemId}
          />

          {/* Discord-style thread view: each thread is an isolated context,
              messages render as left-aligned avatar-led cards, TensorFold marked "APP". */}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <MessageCardView
              items={conversationCards}
              emptyTitle={selectedItem ? 'No messages in this thread yet' : 'Pick a thread'}
              emptyDetail={selectedItem ? 'Ask TensorFold to draft a reply below — source messages and TensorFold\'s draft appear here as cards.' : 'Select a thread on the left to view its messages. Each thread keeps its own context.'}
              onApprove={approveDraft}
              onEdit={editDraft}
              onDiscard={discardDraft}
              onRedraft={redraftDraft}
            />

            <Composer
              taskText={taskText}
              setTaskText={setTaskText}
              canDirectReply={canDirectReply}
              selectedItem={selectedItem}
              selectedSource={selectedSource}
              quickActions={quickActions}
              taskBusy={taskBusy}
              sendingReply={sendingReply}
              onSendReply={sendReply}
              onStartTask={startTask}
              onDraftTask={draftTask}
            />
          </div>
        </div>
      </main>
    </div>
  )
}
