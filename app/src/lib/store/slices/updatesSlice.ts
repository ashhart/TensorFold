import { api } from '../../ipc'
import type { Slice } from '../types'
import type { AppUpdateStatus } from '../../types'

export const updatesSlice: Slice = (set, get) => ({
  updateStatus: null as AppUpdateStatus | null,
  // Update/changelog modal. `updateModalOpen` is the explicit UI gate;
  // `updateModalSeen` records release notes opened explicitly by the user.
  updateModalOpen: false,
  updateInstallPending: false,
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
  async installAppUpdate() {
    if (get().updateInstallPending || get().updateStatus?.state === 'installing') return
    // Cover the host's asynchronous signature/installation preflight before it
    // can publish installing; only an explicit install action opens this modal.
    set({ updateInstallPending: true, updateModalOpen: true })
    try { await api.installAppUpdate() }
    finally { set({ updateInstallPending: false }) }
  },
  applyUpdateStatus(s) {
    set({ updateStatus: s })
  },
  openUpdateModal() {
    set({ updateModalOpen: true })
  },
  closeUpdateModal() {
    if (get().updateInstallPending || get().updateStatus?.state === 'installing') return
    const seen = get().updateStatus?.latestVersion ?? get().updateModalSeen
    try { if (seen) localStorage.setItem('vontra.updateSeen', seen) } catch { /* ignore */ }
    set({ updateModalOpen: false, updateModalSeen: seen })
  },
})
