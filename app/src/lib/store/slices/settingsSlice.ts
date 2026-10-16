import { api } from '../../ipc'
import { resolveTheme } from '../shared/theme'
import type { Slice } from '../types'
import type { Settings, SecurityPurgeResult } from '../../types'

export const settingsSlice: Slice = (set, get) => ({
  settings: null as Settings | null,

  cycleApprovalMode() {
    const s = get().settings
    if (!s) return
    const order: Settings['approvalMode'][] = ['suggest', 'auto', 'readonly']
    const next = order[(order.indexOf(s.approvalMode) + 1) % order.length]
    get().saveSettings({ ...s, approvalMode: next })
  },

  async saveSettings(s, opts) {
    try {
      const saved = await api.saveSettings(s)
      // Confirm every successful *explicit* save so settings edits never feel like a no-op (#48).
      // Silent saves (launch / model switch persist the active provider as a side-effect — the
      // user never touched Settings, so a "Settings saved" toast would be misleading).
      set({ settings: saved, resolvedTheme: resolveTheme(saved), ...(opts?.silent ? {} : { notice: { kind: 'info', message: 'Settings saved' } }) })
    } catch (e) {
      // Never let a persistence failure fail silently — surface it.
      set({ notice: { kind: 'error', message: `Couldn't save settings — ${String((e as any)?.message ?? e)}` } })
      throw e
    }
  },
  async revokeApprovalRule(id) {
    const saved = await api.revokeApprovalRule(id)
    set({ settings: saved, resolvedTheme: resolveTheme(saved) })
  },
  async purgeLocalMemory(): Promise<SecurityPurgeResult> {
    const result = await api.purgeLocalMemory()
    const [settings, threads, projects, rooms, tickets, jobs, automations, archives] = await Promise.all([
      api.getSettings(),
      api.getThreads(),
      api.getProjects(),
      api.getRooms(),
      api.getTickets(),
      api.getJobs(),
      api.getAutomations(),
      api.getArchives(),
    ])
    set({
      settings,
      resolvedTheme: resolveTheme(settings),
      threads,
      projects,
      rooms,
      tickets,
      jobs,
      automations,
      archives,
      status: {},
      approvals: {},
      roomStatus: {},
      roomLive: {},
      view: { name: 'home' },
    })
    return result
  },
})
