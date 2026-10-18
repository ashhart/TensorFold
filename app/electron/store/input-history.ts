import { promises as fs } from 'node:fs'
import { vontraPath } from './paths'
import { readProtectedJson, writeProtectedJson } from './write-infra'

interface HistoryClear { revision: number; before: number }
interface HistoryClears { entries: Record<string, HistoryClear> }
const chains = new Map<string, Promise<unknown>>()
function validate(value: unknown): HistoryClears {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => key !== 'entries')) throw new Error('Input history preferences are unavailable.')
  const entries = (value as HistoryClears).entries
  if (!entries || typeof entries !== 'object' || Array.isArray(entries) || Object.keys(entries).length > 512) throw new Error('Input history preferences are invalid.')
  for (const [key, row] of Object.entries(entries)) if (!/^[a-f0-9]{64}$/.test(key) || !row || typeof row !== 'object' || Array.isArray(row) || Object.keys(row).some(field => !['revision', 'before'].includes(field)) || !Number.isSafeInteger(row.revision) || row.revision < 0 || !Number.isSafeInteger(row.before) || row.before < 0) throw new Error('Input history preferences are invalid.')
  return { entries }
}
function transaction<T>(profile: string, action: (file: string) => Promise<T>): Promise<T> {
  const file = vontraPath('state', 'input-history.json')
  if (vontraPath() !== profile) return Promise.reject(new Error('The active profile changed. Reopen input history.'))
  const next = (chains.get(file) ?? Promise.resolve()).catch(() => {}).then(() => action(file))
  chains.set(file, next)
  void next.catch(() => {}).finally(() => { if (chains.get(file) === next) chains.delete(file) })
  return next
}
async function read(file: string): Promise<HistoryClears> {
  const stat = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!stat) return { entries: {} }
  if (!stat.isFile() || stat.size > 256_000) throw new Error('Input history storage is unavailable.')
  return validate(await readProtectedJson<unknown>(file, undefined, false, true))
}
export function inputHistoryClearState(profile: string, key: string, current: () => void): Promise<HistoryClear> {
  return transaction(profile, async file => { current(); const data = await read(file); current(); return data.entries[key] ?? { revision: 0, before: 0 } })
}
/** Only a cutoff is persisted: prompt text remains exclusively in existing protected chats. */
export function clearInputHistoryState(profile: string, key: string, expectedRevision: number, current: () => void): Promise<HistoryClear> {
  return transaction(profile, async file => {
    current(); const data = await read(file); current()
    const previous = data.entries[key] ?? { revision: 0, before: 0 }
    if (expectedRevision !== previous.revision) throw new Error('Input history changed. Refresh it before clearing.')
    if (previous.revision === Number.MAX_SAFE_INTEGER) throw new Error('Input history revision limit reached.')
    if (!(key in data.entries) && Object.keys(data.entries).length >= 512) throw new Error('Input history preference limit reached; no existing choices were removed.')
    const next = { revision: previous.revision + 1, before: Date.now() }
    const payload = { entries: { ...data.entries, [key]: next } }
    await writeProtectedJson(file, payload, { isEmpty: false, prepareData: () => { current(); return payload }, validateCommit: current })
    current(); return next
  })
}
