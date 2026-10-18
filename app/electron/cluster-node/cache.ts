import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import type { ClusterModel } from '../../shared/cluster-workspace'
import { validateClusterModel } from '../../shared/cluster-validation'
import type { NodeModelInfo } from './types'

export async function boundedJson(file: string, limit = 1_048_576): Promise<Record<string, any>> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NONBLOCK)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.size > limit) throw new Error('Model metadata exceeds its bounded file limit.')
    const body = await handle.readFile()
    if (body.length > limit) throw new Error('Model metadata exceeds its bounded file limit.')
    const value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(body))
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Model metadata must be an object.')
    return value
  } finally { await handle.close() }
}
export async function cachedModelPath(cacheDir: string, input: ClusterModel): Promise<string | null> {
  const model = validateClusterModel(input), root = path.resolve(cacheDir)
  const rootStat = await fs.lstat(root).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!rootStat) return null
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) throw new Error('The approved cache root must be a real directory.')
  const snapshot = path.join(root, `models--${model.repo.replace('/', '--')}`, 'snapshots', model.revision)
  let current = root
  for (const piece of path.relative(root, snapshot).split(path.sep)) {
    current = path.join(current, piece)
    const stat = await fs.lstat(current).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (!stat) return null
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('The immutable model revision namespace must contain real directories.')
  }
  const realRoot = await fs.realpath(root), real = await fs.realpath(snapshot)
  if (!real.startsWith(realRoot + path.sep)) throw new Error('The model revision is outside its approved cache.')
  return snapshot
}
export async function readCachedConfig(cacheDir: string, model: ClusterModel): Promise<{ directory: string; config: Record<string, any> } | null> {
  const directory = await cachedModelPath(cacheDir, model)
  if (!directory) return null
  const configFile = path.join(directory, 'config.json'), real = await fs.realpath(configFile)
  const root = await fs.realpath(cacheDir)
  if (!real.startsWith(root + path.sep)) throw new Error('Model metadata resolves outside the approved cache.')
  return { directory, config: await boundedJson(configFile) }
}
export async function cachedModels(cacheDir: string, modelTypes: Set<string>): Promise<NodeModelInfo[]> {
  const folders = await fs.readdir(cacheDir, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return []; throw error })
  if (folders.length > 2048) throw new Error('Narrow the approved model cache before inspecting this node.')
  const models: NodeModelInfo[] = []
  for (const folder of folders) {
    if (!folder.isDirectory() || folder.isSymbolicLink() || !folder.name.startsWith('models--')) continue
    const parts = folder.name.slice(8).split('--')
    if (parts.length !== 2) continue
    const repo = parts.join('/')
    const snapshots = await fs.readdir(path.join(cacheDir, folder.name, 'snapshots'), { withFileTypes: true }).catch(() => [])
    if (snapshots.length > 128) continue
    for (const entry of snapshots) {
      if (!entry.isDirectory() || entry.isSymbolicLink() || !/^[a-f0-9]{40}$/.test(entry.name)) continue
      try {
        const model = validateClusterModel({ repo, revision: entry.name }), record = await readCachedConfig(cacheDir, model)
        if (!record || !modelTypes.has(record.config.model_type)) continue
        models.push({ ...model, modelType: record.config.model_type, weightsBytes: null, memoryFloorBytes: null })
        if (models.length >= 64) return models
      } catch { /* An invalid cache entry never becomes an advertised usable model. */ }
    }
  }
  return models
}
export function quantizationFormat(config: Record<string, any>): string | null {
  const q = config.quantization
  if (!q || typeof q !== 'object' || !Number.isInteger(q.bits) || !Number.isInteger(q.group_size)) return null
  return `mlx-q${q.bits}g${q.group_size}`
}

/** Cache symlinks may point at HF blobs, but never outside this explicitly approved root. */
export async function validateCachedFiles(cacheDir: string, directory: string): Promise<void> {
  const root = await fs.realpath(cacheDir)
  let seen = 0, weights = 0, tokenizer = false
  const walk = async (folder: string, depth: number): Promise<void> => {
    if (depth > 5) throw new Error('The cached checkpoint exceeds its directory depth limit.')
    const entries = await fs.readdir(folder, { withFileTypes: true })
    for (const entry of entries) {
      if (++seen > 8192) throw new Error('The cached checkpoint exceeds its file inventory limit.')
      const file = path.join(folder, entry.name), real = await fs.realpath(file)
      if (!real.startsWith(root + path.sep)) throw new Error('A checkpoint file resolves outside its approved cache.')
      const stat = await fs.stat(file)
      if (entry.isDirectory() && !entry.isSymbolicLink()) await walk(file, depth + 1)
      else {
        if (!stat.isFile() || entry.isSymbolicLink() && stat.isDirectory()) throw new Error('Checkpoint entries must be bounded regular files or in-cache HF blob links.')
        if (entry.name.endsWith('.safetensors') && stat.size > 0) weights++
        if (entry.name === 'tokenizer.json' && stat.size > 0) tokenizer = true
      }
    }
  }
  await walk(directory, 0)
  if (!weights || !tokenizer) throw new Error('The immutable checkpoint remains partial; weights and tokenizer metadata are required.')
}
