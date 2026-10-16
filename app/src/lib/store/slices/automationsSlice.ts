import { api } from '../../ipc'
import { enqueueAutomationUpdate } from '../shared/queue'
import type { Slice } from '../types'
import type { Automation } from '../../types'

export const automationsSlice: Slice = (set, get) => ({
  automations: [] as Automation[],

  async createAutomation(a) {
    const created = await api.createAutomation(a)
    set((st) => ({ automations: [created, ...st.automations] }))
    return created
  },
  async updateAutomation(id, patch) {
    return enqueueAutomationUpdate(id, async () => {
      // Host validation may reject stale provider/policy configuration. Commit only the
      // authoritative reply so a failed optimistic patch cannot remain visible, and serialize
      // per automation so slower responses cannot overwrite a later edit.
      const updated = await api.updateAutomation(id, patch)
      set((st) => ({ automations: st.automations.map((x) => (x.id === id ? updated : x)) }))
    })
  },
  async deleteAutomation(id) {
    set((st) => ({ automations: st.automations.filter((x) => x.id !== id) }))
    await api.deleteAutomation(id)
  },
  async runAutomation(id) {
    await api.runAutomation(id)
  },
  applyAutomationUpdate(a) {
    if (a.archivedAt) {
      set((st) => ({ automations: st.automations.filter((automation) => automation.id !== a.id) }))
      return
    }
    set((st) => ({ automations: [a, ...st.automations.filter((x) => x.id !== a.id)].sort((x, y) => y.updatedAt - x.updatedAt) }))
  },
})
