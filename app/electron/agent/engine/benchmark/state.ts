import type { ModelBenchmarkReceipt, ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'
import { isRepoId } from '../paths'
import { benchmarkCells, benchmarkProtocolComplete } from './receipt'

/** Pick and validate every persisted renderer field rather than returning arbitrary stored data. */
export function savedBenchmarkSnapshot(raw: any, receipt: ModelBenchmarkReceipt | null): ModelBenchmarkSnapshot {
  const invalid = (): never => { throw new Error('Local benchmark history is invalid.') }
  const keys = ['id', 'model', 'serial', 'state', 'revision', 'startedAt', 'finishedAt', 'progress', 'cells', 'receiptAvailable', 'protocolComplete', 'previousModel', 'restored', 'restoreError', 'error', 'storageError', 'warnings', 'saved', 'metadata']
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || Object.keys(raw).some(key => !keys.includes(key))) invalid()
  if (typeof raw.id !== 'string' || !/^[A-Za-z0-9_-]{16,80}$/.test(raw.id) || !isRepoId(raw.model) || typeof raw.serial !== 'boolean' || !['complete', 'cancelled', 'failed'].includes(raw.state) || !Number.isSafeInteger(raw.revision) || raw.revision < 0 || !Number.isFinite(raw.startedAt) || !Number.isFinite(raw.finishedAt) || raw.finishedAt < raw.startedAt || typeof raw.saved !== 'boolean') invalid()
  if (raw.previousModel !== null && !isRepoId(raw.previousModel) || ![null, true, false].includes(raw.restored)) invalid()
  const message = (value: unknown): string | null => {
    if (value === null) return null
    if (typeof value !== 'string' || value.length > 300 || /[\r\n\0/@\\]|(?:https?:|\/Users|\/home)/i.test(value)) return invalid()
    return value
  }
  const progress = raw.progress
  if (!progress || typeof progress !== 'object' || Object.keys(progress).sort().join(',') !== 'completedRequests,fixture,phase,repeat,temperature,totalRequests' || progress.totalRequests !== 24 || progress.completedRequests !== null && (!Number.isInteger(progress.completedRequests) || progress.completedRequests < 0 || progress.completedRequests > 24) || ![null, 'code', 'chat'].includes(progress.fixture) || ![null, 0, 1].includes(progress.temperature) || ![null, 'loading', 'warmup', 'measured'].includes(progress.phase) || progress.repeat !== null && (!Number.isInteger(progress.repeat) || progress.repeat < -1 || progress.repeat > 4)) invalid()
  if (!Array.isArray(raw.warnings) || raw.warnings.length > 10) invalid()
  return { id: raw.id, model: raw.model, serial: raw.serial, state: raw.state, revision: raw.revision, startedAt: raw.startedAt, finishedAt: raw.finishedAt, progress: structuredClone(progress), cells: benchmarkCells(receipt ?? undefined), receiptAvailable: receipt !== null, protocolComplete: receipt ? benchmarkProtocolComplete(receipt) : false, previousModel: raw.previousModel, restored: raw.restored, restoreError: message(raw.restoreError), error: message(raw.error), storageError: message(raw.storageError), warnings: raw.warnings.map((value: unknown) => message(value) ?? invalid()), saved: raw.saved, metadata: receipt ? structuredClone({ runtime: receipt.runtime, model: receipt.model, hardware: receipt.hardware, settings: receipt.settings }) : null }
}
