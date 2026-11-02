import { promises as fs } from 'node:fs'
import { MicrosoftMailService, type MicrosoftMailState } from '../../agent/microsoft-mail'
import { readProtectedJson, writeProtectedJson } from '../../store/write-infra'
import { vontraPath } from '../../store/paths'
import { store } from '../state'

let instance: MicrosoftMailService | undefined
export function microsoftMailService(): MicrosoftMailService {
  if (instance) return instance
  const file = vontraPath('state', 'microsoft-mail.protected.json')
  instance = new MicrosoftMailService({
    assertSourceCurrent(source) {
      const current = store.settings.email.sources.find(item => item.id === source.id)
      if (!current?.enabled || current.kind !== 'microsoft_graph' || current.user !== source.user || current.microsoftRevision !== source.microsoftRevision) throw new Error('The selected Microsoft account changed. Review the connection again.')
    },
    store: {
    async read() {
      let exists = false
      try {
        const stat = await fs.lstat(file)
        if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || (process.getuid && stat.uid !== process.getuid())) throw new Error('Microsoft connection storage has an unsafe type or owner.')
        exists = true
      } catch (error: any) { if (error.code !== 'ENOENT') throw error }
      const state = await readProtectedJson<MicrosoftMailState | null>(file, null, false, true)
      if (exists && state === null) throw new Error('Microsoft connection storage could not be unlocked. Restore access to the protected account store.')
      return state
    },
    write: (state, validateCommit) => writeProtectedJson(file, state, { validateCommit }),
  } })
  return instance
}
