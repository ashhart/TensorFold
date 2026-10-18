import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { writeProtectedJson } from '../../../store/write-infra'
import { isProtectedJsonEnvelope, revealJsonPayload } from '../../../store/app-crypto'
import type { ModelBenchmarkReceipt, ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'
import { BENCHMARK_RECEIPT_MAX_BYTES, parseBenchmarkReceipt } from './receipt'
import { savedBenchmarkSnapshot } from './state'
import { strictBenchmarkJson } from './strict-json'

export interface SavedModelBenchmark { snapshot: ModelBenchmarkSnapshot; receipt: ModelBenchmarkReceipt | null }
export interface ModelBenchmarkStore { list(): Promise<SavedModelBenchmark[]>; save(record: SavedModelBenchmark): Promise<void> }
function privateStat(stat: Awaited<ReturnType<typeof fs.lstat>>, directory: boolean): void {
  if (stat.isSymbolicLink() || (directory ? !stat.isDirectory() : !stat.isFile() || stat.nlink !== 1)) throw new Error('Benchmark storage has an unsafe file type.')
  if (process.getuid && Number(stat.uid) !== process.getuid() || (Number(stat.mode) & 0o077) !== 0) throw new Error('Benchmark storage has an unsafe owner or permissions.')
}
export async function benchmarkDirectory(appHome: string): Promise<string> {
  const root = path.join(appHome, 'benchmarks')
  await fs.mkdir(root, { recursive: true, mode: 0o700 })
  privateStat(await fs.lstat(root), true)
  return root
}
export async function benchmarkWorkspace(appHome: string): Promise<{ directory: string; receiptPath: string; remove(): Promise<void> }> {
  const root = await benchmarkDirectory(appHome)
  const directory = path.join(root, `.run-${randomUUID()}`)
  await fs.mkdir(directory, { mode: 0o700 })
  return { directory, receiptPath: path.join(directory, 'receipt.json'), remove: () => fs.rm(directory, { recursive: true, force: true }) }
}
export async function readBenchmarkReceiptFile(file: string, expected: { model: string; serial: boolean }): Promise<ModelBenchmarkReceipt | null> {
  const raw = await readPrivateFile(file, BENCHMARK_RECEIPT_MAX_BYTES)
  return raw === null ? null : parseBenchmarkReceipt(raw, expected)
}
async function readPrivateFile(file: string, maxBytes: number): Promise<string | null> {
  let handle: Awaited<ReturnType<typeof fs.open>>
  try { handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW) } catch (error: any) {
    if (error.code === 'ENOENT') return null
    throw new Error('The local benchmark receipt could not be read safely.')
  }
  try {
    const stat = await handle.stat(); privateStat(stat, false)
    if (stat.size < 1 || stat.size > maxBytes) throw new Error('The local benchmark file exceeded its size limit.')
    const chunks: Buffer[] = []; let size = 0
    for (;;) {
      const buffer = Buffer.alloc(Math.min(64 * 1024, maxBytes + 1 - size))
      const { bytesRead } = await handle.read(buffer)
      if (!bytesRead) break
      size += bytesRead
      if (size > maxBytes) throw new Error('The local benchmark file exceeded its size limit.')
      chunks.push(buffer.subarray(0, bytesRead))
    }
    try { return new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks, size)) } catch { throw new Error('The local benchmark file is not valid UTF-8.') }
  } finally { await handle.close() }
}
/** The existing profile/external-key crypto policy, never a new credential store or Keychain. */
export function createModelBenchmarkStore(appHome: string): ModelBenchmarkStore {
  let queue: Promise<unknown> = Promise.resolve()
  const load = async (): Promise<SavedModelBenchmark[]> => {
    const root = await benchmarkDirectory(appHome)
    const file = path.join(root, 'history.protected.json')
    const bytes = await readPrivateFile(file, 16 * BENCHMARK_RECEIPT_MAX_BYTES)
    if (bytes === null) return []
    const raw = strictBenchmarkJson(bytes)
    if (!isProtectedJsonEnvelope(raw)) throw new Error('Benchmark history must use the protected profile store.')
    const records = revealJsonPayload<unknown>(raw, null)
    if (!Array.isArray(records) || records.length > 50) throw new Error('Benchmark history could not be unlocked or validated.')
    return records.map(record => {
      if (!record || typeof record !== 'object' || !record.snapshot || typeof record.snapshot.id !== 'string' || !/^[A-Za-z0-9_-]{16,80}$/.test(record.snapshot.id) || !['complete', 'cancelled', 'failed'].includes(record.snapshot.state)) throw new Error('Benchmark history is invalid.')
      const receipt = record.receipt === null ? null : parseBenchmarkReceipt(JSON.stringify(record.receipt), { model: record.snapshot.model, serial: record.snapshot.serial })
      return { snapshot: savedBenchmarkSnapshot(record.snapshot, receipt), receipt }
    })
  }
  return {
    list: () => queue.then(load),
    save(record) {
      const next = queue.then(async () => {
        const receipt = record.receipt ? parseBenchmarkReceipt(JSON.stringify(record.receipt), { model: record.snapshot.model, serial: record.snapshot.serial }) : null
        const safe = { snapshot: savedBenchmarkSnapshot(record.snapshot, receipt), receipt }
        const records = await load()
        const updated = [safe, ...records.filter(item => item.snapshot.id !== record.snapshot.id)].slice(0, 50)
        const root = await benchmarkDirectory(appHome)
        await writeProtectedJson(path.join(root, 'history.protected.json'), updated)
      })
      queue = next.catch(() => {})
      return next
    },
  }
}
