import { api } from './ipc'
import { createComposerDraftRouting } from './composer-draft-routing'
// A paired browser never reads or writes another device's local drafts through fallback IPC.
export const desktopDraftPort = api.platform === 'darwin' || api.platform === 'win32' || api.platform === 'linux' ? createComposerDraftRouting(api) : null
