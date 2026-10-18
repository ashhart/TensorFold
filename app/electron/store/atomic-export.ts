import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'

/** Host-selected regular destination only; a failed or stale write preserves existing bytes. */
export async function atomicExport(destination: string, bytes: Uint8Array, maxBytes: number, assertCurrent: () => void): Promise<void> {
  if (bytes.byteLength > maxBytes) throw new Error('This export exceeds its size limit.')
  const stat = async () => { try { return await fs.lstat(destination) } catch (error: any) { if (error.code !== 'ENOENT') throw error; return null } }
  const previous = await stat()
  const regular = (value: import('node:fs').Stats) => value.isFile() && !value.isSymbolicLink() && value.nlink === 1
  if (previous && !regular(previous)) throw new Error('Choose a regular export file; linked files are unsupported.')
  const temp = path.join(path.dirname(destination), `.tensorfold-export-${randomUUID()}.tmp`)
  let file: import('node:fs/promises').FileHandle | null = null
  try {
    file = await fs.open(temp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600)
    await file.writeFile(bytes); await file.datasync(); await file.close(); file = null
    const current = await stat()
    if (previous ? !current || !regular(current) || current.ino !== previous.ino || current.dev !== previous.dev || current.mtimeMs !== previous.mtimeMs || current.size !== previous.size : current !== null) throw new Error('The export destination changed; choose it again.')
    assertCurrent()
    await fs.rename(temp, destination)
  } finally { await file?.close().catch(() => {}); await fs.unlink(temp).catch(() => {}) }
}
