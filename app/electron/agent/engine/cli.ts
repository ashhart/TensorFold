import { execFile } from 'node:child_process'
import type { EngineCapabilities } from '../types'
import type { EngineBinary } from './paths'

// Thin wrappers over the engine's CPU-only subcommands (--version, models, info, capabilities --json)
// and parsers for their text, matching zig/src/cli/{models,info}.zig and server/cli.zig in 1.0.7.

export interface CliResult {
  code: number
  stdout: string
  stderr: string
}

export async function runEngineCli(binary: EngineBinary, args: string[], env: NodeJS.ProcessEnv, timeoutMs = 30_000): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    execFile(binary.command, [...binary.args, ...args], { env, timeout: timeoutMs, maxBuffer: 4 * 1024 * 1024, encoding: 'utf8' }, (error, stdout, stderr) => {
      const err = error as (NodeJS.ErrnoException & { code?: number | string; killed?: boolean }) | null
      if (err && typeof err.code !== 'number') {
        reject(new Error(err.killed ? `tensorfold-native ${args[0]} timed out.` : `Could not run tensorfold-native: ${err.message}`))
        return
      }
      resolve({ code: err ? Number(err.code) : 0, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
    })
  })
}

export function parseVersion(text: string): string | null {
  return /^tensorfold-native\s+(\d+\.\d+\.\d+)\s*$/m.exec(text)?.[1] ?? null
}

export interface ListedModel {
  repo: string
  family: string
  modelType: string
}

// "Org/Name: Family title (model_type)", one per servable checkpoint; anything else is a notice.
export function parseModelsOutput(text: string): ListedModel[] {
  const out: ListedModel[] = []
  for (const line of text.split(/\r?\n/)) {
    const match = /^([A-Za-z0-9._-]+\/[A-Za-z0-9._-]+): (.+) \(([A-Za-z0-9_.-]+)\)$/.exec(line.trim())
    if (match) out.push({ repo: match[1], family: match[2], modelType: match[3] })
  }
  return out
}

export interface ModelInfo {
  family: string
  modelType: string
  format: string | null
  contextTokens: number | null
  weightsBytes: number
  memoryFloorBytes: number
}

const GIB = 1024 ** 3

export function parseInfoOutput(text: string): ModelInfo {
  const field = (name: string) => new RegExp(`^${name}: (.+)$`, 'm').exec(text)?.[1]?.trim() ?? null
  const family = /^(.+) \(([A-Za-z0-9_.-]+)\)$/.exec(field('family') ?? '')
  const weights = /\((\d+) bytes\)/.exec(field('weights') ?? '')
  const floor = /^([\d.]+) GiB \(weights plus a ([\d.]+) GiB runtime margin\)$/.exec(field('memory floor') ?? '')
  if (!family || !weights || !floor) throw new Error(text.trim().split(/\r?\n/)[0] || 'tensorfold-native info printed nothing.')
  const weightsBytes = Number(weights[1])
  const context = Number(field('context'))
  return {
    family: family[1],
    modelType: family[2],
    format: field('format'),
    contextTokens: Number.isFinite(context) && context > 0 ? context : null,
    weightsBytes,
    // Recompute from exact bytes: weights plus the printed runtime margin.
    memoryFloorBytes: weightsBytes + Math.round(Number(floor[2]) * GIB),
  }
}

export function parseCapabilities(text: string): EngineCapabilities {
  let doc: any
  try { doc = JSON.parse(text) } catch { throw new Error('tensorfold-native capabilities --json did not print JSON.') }
  if (!doc || typeof doc !== 'object' || doc.engine !== 'zig' || typeof doc.version !== 'string') throw new Error('Unexpected capabilities document.')
  const families: Record<string, string[]> = {}
  for (const [modelType, formats] of Object.entries(doc.families ?? {})) {
    families[modelType] = Array.isArray(formats) ? formats.filter((f): f is string => typeof f === 'string') : []
  }
  return {
    version: doc.version,
    chip: typeof doc.chip === 'string' ? doc.chip : null,
    backends: Array.isArray(doc.backends) ? doc.backends.filter((b: unknown): b is string => typeof b === 'string') : [],
    families,
    flags: Object.keys(doc.serve?.flags ?? {}),
    env: Array.isArray(doc.env) ? doc.env.filter((e: unknown): e is string => typeof e === 'string') : [],
  }
}

// MLX affine quantization as the engine's format names spell it: mlx-q4g64, mlx-q6g32.
export function formatName(config: { quantization?: { bits?: unknown; group_size?: unknown } | null } | null | undefined): string | null {
  const q = config?.quantization
  if (!q || typeof q !== 'object') return null
  const bits = Number(q.bits)
  const group = q.group_size == null ? 32 : Number(q.group_size)
  if (!Number.isInteger(bits) || !Number.isInteger(group)) return null
  return `mlx-q${bits}g${group}`
}

export function describeFormat(format: string | null): string {
  const match = /^mlx-q(\d+)g(\d+)$/.exec(format ?? '')
  return match ? `${match[1]}-bit, group ${match[2]}` : format ? format : 'full precision'
}
