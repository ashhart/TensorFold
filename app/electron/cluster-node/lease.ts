import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { privateDirectory, privateRead } from './config'

/** A live or ambiguous owner is never displaced; a dead PID only releases its unchanged lock record. */
export async function acquireNodeLease(dataDir: string): Promise<{ release(): Promise<void> }> {
  await privateDirectory(dataDir)
  const file = path.join(dataDir, 'node.lock'), nonce = randomUUID()
  const claim = async () => fs.open(file, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600)
  let handle
  try { handle = await claim() }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
    const before = await fs.lstat(file), bytes = await privateRead(file, 1024)
    const owner = JSON.parse(bytes.toString('utf8'))
    if (owner?.schema !== 1 || !Number.isSafeInteger(owner.pid) || owner.pid <= 0 || typeof owner.nonce !== 'string') throw new Error('The node ownership lock needs manual review.')
    try { process.kill(owner.pid, 0); throw new Error('Another live process owns this node data directory.') }
    catch (probe) { if ((probe as NodeJS.ErrnoException).code !== 'ESRCH') throw probe }
    const after = await fs.lstat(file), latest = await privateRead(file, 1024)
    if (before.ino !== after.ino || before.dev !== after.dev || !bytes.equals(latest)) throw new Error('The node ownership lock changed; try again.')
    await fs.unlink(file)
    handle = await claim()
  }
  try { await handle.writeFile(JSON.stringify({ schema: 1, pid: process.pid, nonce })); await handle.sync() }
  catch (error) { await handle.close(); await fs.unlink(file).catch(() => {}); throw error }
  let released = false
  return { release: async () => {
    if (released) return; released = true
    await handle.close()
    const current = await privateRead(file, 1024).catch(() => null)
    if (current && JSON.parse(current.toString('utf8')).nonce === nonce) await fs.unlink(file)
  } }
}
