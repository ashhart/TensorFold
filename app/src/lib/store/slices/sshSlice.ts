import { api } from '../../ipc'
import type { Slice } from '../types'
import type { SshHost, SshConnectionState, SshSecretPrompt, SshTrustPrompt } from '../../types'

export const sshSlice: Slice = (set, get) => ({
  sshHosts: [] as SshHost[],
  sshStatus: {} as Record<string, SshConnectionState>,
  sshSecretPrompt: null as SshSecretPrompt | null,
  sshTrustPrompt: null as SshTrustPrompt | null,

  async refreshSshHosts() {
    const sshHosts = await api.listSshHosts()
    set({ sshHosts })
  },
  async connectSshHost(hostId) {
    try {
      const state = await api.connectSshHost(hostId)
      set((st) => ({ sshStatus: { ...st.sshStatus, [hostId]: state } }))
    } catch (e) {
      set((st) => ({ sshStatus: { ...st.sshStatus, [hostId]: { hostId, phase: 'error', lastError: String((e as any)?.message ?? e) } } }))
    }
  },
  async disconnectSshHost(hostId) {
    await api.disconnectSshHost(hostId)
    set((st) => ({ sshStatus: { ...st.sshStatus, [hostId]: { hostId, phase: 'disconnected' } } }))
  },
  async resolveSshSecret(value) {
    const p = get().sshSecretPrompt
    if (!p) return
    set({ sshSecretPrompt: null })
    await api.resolveSshSecret(p.promptId, value)
  },
  async resolveSshTrust(ok) {
    const p = get().sshTrustPrompt
    if (!p) return
    set({ sshTrustPrompt: null })
    await api.resolveSshTrust(p.promptId, ok)
  },
})
