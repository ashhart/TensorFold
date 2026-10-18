import { app } from 'electron'
import { promises as fs } from 'node:fs'
import { GoogleWorkspaceService, type GoogleWorkspacePrivateState } from '../../agent/google-workspace'
import { readProtectedJson, writeProtectedJson } from '../../store/write-infra'
import { vontraPath } from '../../store/paths'
import { openExternalSafely } from './external'

let instance: GoogleWorkspaceService | undefined

/** A separate protected collection keeps OAuth tokens out of Settings and renderer snapshots. */
export function googleWorkspaceService(): GoogleWorkspaceService {
  if (instance) return instance
  const file = vontraPath('state', 'google-workspace.protected.json')
  instance = new GoogleWorkspaceService({
    openExternal: (url) => openExternalSafely(url),
    store: {
      async read() {
        let exists = false
        try {
          const stat = await fs.lstat(file)
          if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || (process.getuid && stat.uid !== process.getuid())) throw new Error('Google connection storage has an unsafe type or owner.')
          exists = true
        } catch (error: any) { if (error.code !== 'ENOENT') throw error }
        const state = await readProtectedJson<GoogleWorkspacePrivateState | null>(file, null, false, true)
        if (exists && state === null) throw new Error('Google connection storage could not be unlocked; reconnect after restoring access to the local credential store.')
        return state
      },
      write: (state, validateCommit) => writeProtectedJson(file, state, { validateCommit }),
    },
  })
  app.once('before-quit', () => { instance?.dispose(); instance = undefined })
  return instance
}
