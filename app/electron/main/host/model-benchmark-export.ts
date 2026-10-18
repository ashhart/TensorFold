import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'

/** Only a native chooser supplies the destination; preserve existing bytes on failed writes. */
export async function saveBenchmarkExport(destination: string, json: string, assertAllowed: () => void): Promise<void> {
  if (Buffer.byteLength(json, 'utf8') > 1_048_576) throw new Error('Benchmark receipt is too large to export.')
  let previous: import('node:fs').Stats | null = null
  try { previous = await fs.lstat(destination) } catch (error: any) { if (error.code !== 'ENOENT') throw error }
  if (previous && (!previous.isFile() || previous.isSymbolicLink() || previous.nlink !== 1)) throw new Error('Choose a regular JSON file; linked destinations are unsupported.')
  const temp = path.join(path.dirname(destination), `.tensorfold-benchmark-${randomUUID()}.tmp`)
  let handle: import('node:fs/promises').FileHandle | null = null
  try {
    handle = await fs.open(temp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600)
    await handle.writeFile(json, 'utf8'); await handle.datasync(); await handle.close(); handle = null
    assertAllowed()
    let current: import('node:fs').Stats | null = null
    try { current = await fs.lstat(destination) } catch (error: any) { if (error.code !== 'ENOENT') throw error }
    if (previous ? !current || !current.isFile() || current.isSymbolicLink() || current.nlink !== 1 || current.ino !== previous.ino || current.dev !== previous.dev || current.mtimeMs !== previous.mtimeMs || current.size !== previous.size : current !== null) throw new Error('The selected export destination changed; choose it again.')
    await fs.rename(temp, destination)
  } finally { await handle?.close().catch(() => {}); await fs.unlink(temp).catch(() => {}) }
}
