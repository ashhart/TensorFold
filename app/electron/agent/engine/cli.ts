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
// Engine 1.3.0 prints one of two priced startup-budget lines, both of which name the floor. The
// conservative Metal plan is fully parenthesised; the CUDA/Metal --context plan is not. "not priced"
// matches neither, so an unpriced budget still falls through to the floor (or is a parse failure).
const STARTUP_BUDGET_RE = /^(\d+(?:\.\d+)?) GiB \(conservative raw\/packed text weights, both prompt planes\/selectors, KV\/state, keepalive and \d+(?:\.\d+)? GiB runtime reserve; trained window\)$/
const STARTUP_BUDGET_AT_CONTEXT_RE = /^(\d+(?:\.\d+)?) GiB at --context \d+: /

export function parseInfoOutput(text: string): ModelInfo {
  const field = (name: string) => new RegExp(`^${name}: (.+)$`, 'm').exec(text)?.[1]?.trim() ?? null
  const family = /^(.+) \(([A-Za-z0-9_.-]+)\)$/.exec(field('family') ?? '')
  const weights = /\((\d+) bytes\)/.exec(field('weights') ?? '')
  const floor = /^([\d.]+) GiB \(weights plus a ([\d.]+) GiB runtime margin\)$/.exec(field('memory floor') ?? '')
  const budgetLine = field('startup budget') ?? ''
  const startup = STARTUP_BUDGET_RE.exec(budgetLine) ?? STARTUP_BUDGET_AT_CONTEXT_RE.exec(budgetLine)
  if (!family || !weights || (!floor && !startup)) throw new Error(text.trim().split(/\r?\n/)[0] || 'tensorfold-native info printed nothing.')
  const weightsBytes = Number(weights[1])
  const context = Number(field('context'))
  const memoryFloorBytes = floor ? weightsBytes + Math.round(Number(floor[2]) * GIB) : Math.round(Number(startup![1]) * GIB)
  if (!Number.isSafeInteger(memoryFloorBytes) || memoryFloorBytes <= 0) throw new Error(text.trim().split(/\r?\n/)[0] || 'tensorfold-native info printed nothing.')
  return {
    family: family[1],
    modelType: family[2],
    format: field('format'),
    contextTokens: Number.isFinite(context) && context > 0 ? context : null,
    weightsBytes,
    // Recompute legacy floor from exact bytes; startup budget already includes weights and runtime reserve.
    memoryFloorBytes,
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

interface CheckpointFormat {
  model_type?: unknown
  quantization?: { bits?: unknown; group_size?: unknown } | null
  quantization_config?: unknown
  compression_config?: unknown
  dtype?: unknown
  torch_dtype?: unknown
  text_config?: CheckpointFormat | null
}

// Format names as advertised by the engine, such as mlx-q4g64 and bf16.
export function formatName(config: CheckpointFormat | null | undefined): string | null {
  const q = config?.quantization
  if (q == null) {
    // The capabilities list also includes BF16 engines for embeddings and OCR.
    if (config?.model_type !== 'spark2_5') return null
    const tower = config?.text_config
    if ([config, tower].some((part) => part?.quantization != null || part?.quantization_config != null || part?.compression_config != null)) return null
    const dtype = tower?.dtype ?? tower?.torch_dtype ?? config?.dtype ?? config?.torch_dtype
    return dtype === 'bfloat16' ? 'bf16' : null
  }
  if (typeof q !== 'object') return null
  const bits = Number(q.bits)
  const group = q.group_size == null ? 32 : Number(q.group_size)
  if (!Number.isInteger(bits) || !Number.isInteger(group)) return null
  return `mlx-q${bits}g${group}`
}

// TFQ3 record files at a checkpoint's root, as each Metal loader that reads them looks for them before turning TFQ3 on:
// glm/tfq3.zig hasRecords (nonmoe or layer_NN) and flashnext/tfq3_files.zig hasRecords (layer_NN or mtp_00).
const TFQ3_RECORDS: Record<string, RegExp> = {
  glm5_next: /^(nonmoe|layer_\d{2})\.tffpw$/,
  qwen4_exp: /^(layer_\d{2}|mtp_00)\.tffpw$/,
}
const ANY_TFQ3_RECORD = /^(nonmoe|layer_\d{2}|mtp_00)\.tffpw$/

// Whether a checkpoint's root file names hold TFQ3 records; pw_v3_tables.bin alone is not a record.
export function hasTfq3Records(modelType: string | null | undefined, names: readonly string[]): boolean {
  const rule = (modelType && TFQ3_RECORDS[modelType]) || ANY_TFQ3_RECORD
  return names.some((name) => rule.test(name))
}

// The format a checkpoint is checked as: tfq3 when its root holds TFQ3 records (config.json keeps a plain MLX header).
export function checkpointFormat(config: CheckpointFormat | null | undefined, modelType: string | null | undefined, names: readonly string[]): string | null {
  return hasTfq3Records(modelType, names) ? 'tfq3' : formatName(config)
}

export function describeFormat(format: string | null): string {
  if (format === 'tfq3') return 'TFQ3'
  const match = /^mlx-q(\d+)g(\d+)$/.exec(format ?? '')
  return match ? `${match[1]}-bit, group ${match[2]}` : format ? format : 'full precision'
}
