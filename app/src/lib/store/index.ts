import { create } from 'zustand'
import { api } from '../ipc'
import { catchUpFromSpine } from '../spine-projector'
import { applyWhatsAppStatusPush } from '../whatsapp-status'
import { reconcileSessionsForOrigin, terminalSessionKey, updateSessionStatus } from '../terminal-sessions'
import type { PersistenceFailure } from '../types'
import type { AppState, GetState, SetState } from './types'
import { acquireInitStarted, acquireSubscriptionsStarted } from './shared/app-actions'
import { resolveTheme } from './shared/theme'
import { mergeOriginBoundRecords, mergeRecordsByFreshest, projectedHostId, projectedProjectHostId } from './shared/proxy-state'
import { mergeThreadLists } from './shared/merge'
import { drainPendingAgentEvents } from './shared/proxy-state'
import { uiSlice } from './slices/uiSlice'
import { viewSlice } from './slices/viewSlice'
import { settingsSlice } from './slices/settingsSlice'
import { threadsSlice } from './slices/threadsSlice'
import { projectsSlice } from './slices/projectsSlice'
import { roomsSlice } from './slices/roomsSlice'
import { ticketsSlice } from './slices/ticketsSlice'
import { automationsSlice } from './slices/automationsSlice'
import { updatesSlice } from './slices/updatesSlice'
import { terminalsSlice } from './slices/terminalsSlice'
import { sshSlice } from './slices/sshSlice'
import { archivesSlice } from './slices/archivesSlice'

// Each slice returns a `Partial<AppState>` (its own fields + actions). Spreading them together
// covers every AppState field, but TypeScript widens the spread to all-optional, so the combined
// object is asserted as AppState. `init` is wired here (not in any slice) because it subscribes
// every IPC listener and loads every collection.
const buildState = (set: SetState, get: GetState): AppState => ({
  ...uiSlice(set, get),
  ...viewSlice(set, get),
  ...settingsSlice(set, get),
  ...threadsSlice(set, get),
  ...projectsSlice(set, get),
  ...roomsSlice(set, get),
  ...ticketsSlice(set, get),
  ...automationsSlice(set, get),
  ...updatesSlice(set, get),
  ...terminalsSlice(set, get),
  ...sshSlice(set, get),
  ...archivesSlice(set, get),

  async init(options) {
    if (options?.retry) {
      if (!get().ready) return
      set({ ready: false })
    } else if (!acquireInitStarted()) return // StrictMode must not double-subscribe.
    const updateStatusAtStart = get().updateStatus
    const newSubscriptions = acquireSubscriptionsStarted()
    if (newSubscriptions) {
      api.onAgentEvent((e) => get().applyEvent(e))
      api.onProjectUpdate((p) => get().applyProjectUpdate(p))
      api.onThreadUpdate((t) => get().applyThreadUpdate(t))
      api.onRoomUpdate((r) => get().applyRoomUpdate(r))
      api.onRoomEvent((e) => get().applyRoomEvent(e))
      api.onTicketUpdate((t) => get().applyTicketUpdate(t))
      api.onJobUpdate((j) => get().applyJobUpdate(j))
      api.onAutomationUpdate((a) => get().applyAutomationUpdate(a))
      api.onUpdateStatus((s) => get().applyUpdateStatus(s))
      api.onOpenThread?.((threadId) => { if (get().threads.some((t) => t.id === threadId)) get().openThread(threadId) })
      // The host persists bridge status before broadcasting it. Mirror that status into the
      // renderer cache without calling saveSettings: otherwise sidebar/readiness surfaces remain
      // stale, while saving here could overwrite an unrelated in-progress Settings draft.
      api.onWhatsAppStatus((status) => set((state) => ({
        settings: state.settings ? applyWhatsAppStatusPush(state.settings, status) : null,
      })))
      api.onPersistenceFailure((failure: PersistenceFailure) => set({
        notice: {
          kind: 'error',
          message: `Local save failed for ${failure.label}. Recent changes may not survive restart. ${failure.message}`,
        },
      }))
      api.onSshStatus((s) => set((st) => ({
        sshStatus: { ...st.sshStatus, [s.hostId]: s },
        ...((s.phase === 'error' || s.phase === 'disconnected') ? {
          sshSecretPrompt: st.sshSecretPrompt?.hostId === s.hostId ? null : st.sshSecretPrompt,
          sshTrustPrompt: st.sshTrustPrompt?.hostId === s.hostId ? null : st.sshTrustPrompt,
        } : {}),
      })))
      api.onSshSecretPrompt((p) => set({ sshSecretPrompt: p }))
      api.onSshTrustPrompt((p) => set({ sshTrustPrompt: p }))
      api.onTerminalSessions(({ projectId, sessions, orchestratorHostId = null }) => set((st) => {
        const terminals = reconcileSessionsForOrigin(st.terminals, projectId, sessions, orchestratorHostId)
        const aggregate = terminals[projectId] ?? []
        const active = st.activeTerminalId[projectId]
        const stillThere = active && aggregate.some((s) => terminalSessionKey(s) === active)
        return {
          terminals,
          activeTerminalId: { ...st.activeTerminalId, [projectId]: stillThere ? active : (aggregate[0] ? terminalSessionKey(aggregate[0]) : null) },
        }
      }))
      api.onTerminalStatus(({ sessionId, status, orchestratorHostId = null }) => set((st) => ({ terminals: updateSessionStatus(st.terminals, sessionId, status, orchestratorHostId) })))
    }
    try {
      // Settle each load independently: one failing IPC call (these match QUIET_IPC so no toast)
      // must not throw out of init() and leave ready:false forever — that wedges App.tsx on a
      // permanent blank screen (`if (!ready) return null`). Fall back to the existing defaults.
      const results = await Promise.allSettled([api.getSettings(), api.getThreads(), api.getProjects(), api.getRooms(), api.getTickets(), api.getJobs(), api.getAutomations(), api.getUpdateStatus(), api.getArchives()])
      const failed = results.some((r) => r.status === 'rejected')
      const cur = get()
      const pick = <T>(r: PromiseSettledResult<T>, fallback: T): T => (r.status === 'fulfilled' ? r.value : fallback)
      const [settings, threads, projects, rooms, tickets, jobs, automations, updateStatus, archives] = [
        pick(results[0] as PromiseSettledResult<typeof cur.settings>, cur.settings),
        pick(results[1] as PromiseSettledResult<typeof cur.threads>, cur.threads),
        pick(results[2] as PromiseSettledResult<typeof cur.projects>, cur.projects),
        pick(results[3] as PromiseSettledResult<typeof cur.rooms>, cur.rooms),
        pick(results[4] as PromiseSettledResult<typeof cur.tickets>, cur.tickets),
        pick(results[5] as PromiseSettledResult<typeof cur.jobs>, cur.jobs),
        pick(results[6] as PromiseSettledResult<typeof cur.automations>, cur.automations),
        pick(results[7] as PromiseSettledResult<typeof cur.updateStatus>, cur.updateStatus),
        pick(results[8] as PromiseSettledResult<typeof cur.archives>, cur.archives),
      ]
      const live = get()
      set({
        settings,
        threads: mergeThreadLists(threads, live.threads),
        projects: mergeOriginBoundRecords(projects, live.projects, projectedProjectHostId),
        rooms: mergeOriginBoundRecords(rooms, live.rooms, projectedHostId),
        tickets: mergeOriginBoundRecords(tickets, live.tickets, projectedHostId),
        jobs: mergeRecordsByFreshest(jobs, live.jobs),
        automations: mergeRecordsByFreshest(automations, live.automations),
        // Live host pushes received during hydration supersede its captured update snapshot.
        updateStatus: live.updateStatus !== updateStatusAtStart ? live.updateStatus : updateStatus,
        archives,
        resolvedTheme: resolveTheme(settings),
        notice: failed ? { kind: 'error', message: 'Some data failed to load — running with partial state.', authored: true } : live.notice,
      })
    } finally {
      // Always render: even on total failure the UI must come up rather than hang on a blank screen.
      set({ ready: true })
    }
    // Remote archives are a separate authenticated projection (active snapshots intentionally
    // omit tombstones). Refresh after settings load without delaying first paint on an offline host.
    void get().refreshArchives().catch(() => {})
    for (const thread of get().threads) {
      const events = drainPendingAgentEvents(thread.id, projectedHostId(thread))
      for (const event of events) get().applyEvent(event)
    }
    api.listSshHosts().then((sshHosts) => set({ sshHosts })).catch(() => {})
    if (newSubscriptions) window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', () => {
      const s = get().settings
      if (s?.theme === 'system') set({ resolvedTheme: resolveTheme(s) })
    })

    // Spine projector (VST-034): the renderer keeps a PERSISTED cursor over the host's
    // ordered event log and gap-fills on boot — plus on every SSE re-baseline (a paired
    // client that was away longer than the live ring buffers, VST-041). A digest of what
    // was missed surfaces as a notice instead of a silent snapshot swap.
    const runSpineCatchUp = () => {
      void catchUpFromSpine({
        readSince: (input) => api.spineReadSince(input),
        advanceCursor: (input) => api.spineAdvanceCursor(input),
        loadCursor: () => {
          try { return Number(localStorage.getItem('vontra.spine.cursor') ?? 0) || 0 } catch { return 0 }
        },
        saveCursor: (sequence) => {
          try { localStorage.setItem('vontra.spine.cursor', String(sequence)) } catch { /* private mode */ }
        },
      }).then((catchUp) => {
        if (!catchUp?.summary) return
        const suffix = catchUp.truncated ? ' (older history was compacted)' : ''
        set({ notice: { kind: 'info', message: `${catchUp.summary}${suffix}` } })
      }).catch(() => {})
    }
    runSpineCatchUp()
    if (newSubscriptions) window.addEventListener('vontra:spine-rebaseline', runSpineCatchUp)
  },
} as AppState)

export const useStore = create<AppState>(buildState)

// Dev-only debug handle: lets local tooling/QA inspect and drive store state
// from the console. Stripped from production builds.
if ((import.meta as any).env?.DEV) (globalThis as any).__store = useStore
