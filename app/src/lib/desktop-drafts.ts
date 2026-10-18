import { api } from './ipc'
// A paired browser never reads or writes another device's local drafts through fallback IPC.
export const desktopDraftPort = api.platform === 'darwin' || api.platform === 'win32' || api.platform === 'linux' ? api : null
