import { api } from '../../ipc'
import type { Slice } from '../types'
import type { AppUpdateStatus } from '../../types'

export const updatesSlice: Slice = (set, get) => ({
  updateStatus: null as AppUpdateStatus | null,
  // Update/changelog modal. `updateModalOpen` is the explicit UI gate;
  // `updateModalSeen` tracks the latest version the user has already viewed
  // the release notes for, so we only auto-pop on a genuinely new release.
  updateModalOpen: false,
  updateModalSeen: (() => { try { return localStorage.getItem('vontra.updateSeen') } catch { return null } })() as string | null,

  async checkForUpdates() {
    if (['downloading', 'ready', 'installing'].includes(get().updateStatus?.state ?? '')) return
    set((st) => ({ updateStatus: st.updateStatus ? { ...st.updateStatus, state: 'checking', error: null } : st.updateStatus }))
    const next = await api.checkForUpdates()
    set({ updateStatus: next })
  },
  async openUpdateDownload() {
    await api.openUpdateDownload()
  },
  async downloadAppUpdate() { await api.downloadAppUpdate() },
  async cancelAppUpdateDownload() { await api.cancelAppUpdateDownload() },
  async installAppUpdate() { await api.installAppUpdate() },
  applyUpdateStatus(s) {
    set({ updateStatus: s })
  },
  openUpdateModal() {
    set({ updateModalOpen: true })
  },
  closeUpdateModal() {
    const seen = get().updateStatus?.latestVersion ?? get().updateModalSeen
    try { if (seen) localStorage.setItem('vontra.updateSeen', seen) } catch { /* ignore */ }
    set({ updateModalOpen: false, updateModalSeen: seen })
  },
})
