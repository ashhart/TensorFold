import type { EngineInstalledModel, EngineStatus, ModelPreflight, SuggestedModel } from './types'

const LIGHTWEIGHT = 'mlx-community/Qwen3.5-2B-MLX-4bit'
const BALANCED = 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
const QUALIFIED_LARGE = 'TensorFold/Qwen3.8-27B-MLX-4bit'
const FLASH_NEXT = 'TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP'
const GIB = 1024 ** 3

export interface RecommendedEngineModel {
  repo: string
  title: string
  reason: string
  /** True only when all files needed to load, including a companion drafter, are installed. */
  installed: boolean
  action: 'load' | 'download'
  memoryBytes: number
}

export interface ModelRecommendation {
  status: 'checking' | 'ready' | 'unavailable' | 'unsupported'
  selected: RecommendedEngineModel | null
  fallback: RecommendedEngineModel | null
  reason: string
  hardwareLabel: string
}

export interface ModelRecommendationInput {
  status: EngineStatus | null | undefined
  suggestions: readonly SuggestedModel[]
  installed: readonly EngineInstalledModel[]
  preflights: Readonly<Record<string, ModelPreflight | null | undefined>>
}

type Proof = { state: 'ready'; installed: boolean; memoryBytes: number } | { state: 'checking' | 'unavailable' }
const positive = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0
const nonnegative = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0

function modelProof(installed: EngineInstalledModel | undefined, preflight: ModelPreflight | null | undefined, limit: number, requiresDrafter: boolean): Proof {
  const installedComplete = installed?.fitsMemory === true && positive(installed.memoryFloorBytes) && installed.memoryFloorBytes <= limit
    && (!requiresDrafter || !!installed.drafter)
    && (!installed.drafter || installed.drafterInstalled === true)
  if (installedComplete) return { state: 'ready', installed: true, memoryBytes: installed.memoryFloorBytes! }
  if (!preflight) return { state: installed?.fitsMemory === false || positive(installed?.memoryFloorBytes) && installed.memoryFloorBytes > limit ? 'unavailable' : 'checking' }
  if (!['ok', 'installed'].includes(preflight.verdict) || !preflight.supported || preflight.formatOk === false || preflight.fitsMemory === false || preflight.fitsDisk === false || preflight.gated !== false) return { state: 'unavailable' }
  if (preflight.formatOk !== true || !preflight.format || preflight.fitsMemory !== true || preflight.fitsDisk !== true
    || !positive(preflight.memoryFloorBytes) || !positive(preflight.totalBytes) || !positive(preflight.memoryLimitBytes)
    || !nonnegative(preflight.freeDiskBytes) || !nonnegative(preflight.neededDiskBytes)
    || requiresDrafter && !preflight.drafter
    || preflight.drafter && !positive(preflight.drafter.totalBytes)) return { state: 'checking' }
  // Recheck numeric evidence against the current device; a saved preflight may
  // have been produced on a device with a different memory limit.
  if (preflight.memoryFloorBytes > Math.min(limit, preflight.memoryLimitBytes) || preflight.neededDiskBytes > preflight.freeDiskBytes) return { state: 'unavailable' }
  // A preflight proves a download can fit. It does not prove that its files or
  // companion drafter are on disk, even when the main repo is already cached.
  return { state: 'ready', installed: false, memoryBytes: preflight.memoryFloorBytes }
}

function hardwareLabel(status: EngineStatus | null | undefined): string {
  const hardware = status?.hardware
  const memory = positive(status?.totalMemoryBytes) ? `${(status.totalMemoryBytes / GIB).toLocaleString('en-US', { maximumFractionDigits: 1 })} GiB memory` : null
  const platform = hardware?.platform === 'darwin' ? 'macOS' : hardware?.platform === 'win32' ? 'Windows' : hardware?.platform === 'linux' ? 'Linux' : hardware?.platform
  const chip = hardware?.chip?.trim() || (platform && hardware?.architecture ? `${platform} on ${hardware.architecture}` : 'Hardware details unavailable')
  return memory ? `${chip} with ${memory}` : chip
}

function modelReason(repo: string, proof: Extract<Proof, { state: 'ready' }>, qualifiedChip: boolean, flashChip: boolean): string {
  const fit = proof.installed ? 'Installed with enough memory to load.' : 'Its format, memory use, and download space have passed the device checks.'
  if (repo === BALANCED) return `A balanced model for everyday work on this Mac. ${fit}`
  if (repo === LIGHTWEIGHT) return `A lightweight choice that leaves more memory for your other apps. ${fit}`
  if (repo === FLASH_NEXT) return flashChip
    ? `The largest model this Mac can hold, qualified on this chip, with built-in multi-token drafting. ${fit}`
    : `This model is qualified on M5 Ultra; this Mac has passed the available compatibility and memory checks. ${fit}`
  if (repo === QUALIFIED_LARGE) return qualifiedChip
    ? `A larger model qualified on this chip, with its companion drafter. ${fit}`
    : `This model is qualified on M5 Max and M3 Ultra; this Mac has passed the available compatibility and memory checks. ${fit}`
  return fit
}

/** Recommend only curated models with live compatibility evidence; this never starts a download. */
export function recommendEngineModel(input: ModelRecommendationInput): ModelRecommendation {
  const { status, suggestions, installed, preflights } = input
  const label = hardwareLabel(status)
  const result = (state: ModelRecommendation['status'], reason: string): ModelRecommendation => ({ status: state, selected: null, fallback: null, reason, hardwareLabel: label })
  const hardware = status?.hardware
  if (!status || !hardware?.platform || !hardware.architecture) return result('checking', 'Reading device details before recommending a model.')
  if (hardware.platform !== 'darwin' || hardware.architecture !== 'arm64') return result('unsupported', 'The bundled TensorFold engine requires macOS on Apple silicon; this device cannot run this build.')
  if (!status.available) return result('unavailable', 'The built-in TensorFold engine is unavailable in this build.')
  if (!positive(status.memoryLimitBytes)) return result('checking', 'Checking how much memory this device can use for a model.')
  if (suggestions.length === 0) return result('unavailable', 'No curated models are available to check in this build.')
  const qualifiedChip = /\b(?:M5\s+Max|M3\s+Ultra)\b/i.test(hardware.chip ?? '')
  const flashChip = /\bM5\s+Ultra\b/i.test(hardware.chip ?? '')
  // Flash Next and the 27B lead only on their qualified chips; elsewhere they stay available but rank last.
  const rank = (repo: string): number => repo === FLASH_NEXT ? (flashChip ? -1 : 5)
    : repo === QUALIFIED_LARGE && qualifiedChip ? 0 : repo === BALANCED ? 1 : repo === LIGHTWEIGHT ? 2 : repo === QUALIFIED_LARGE ? 4 : 3
  const candidates = suggestions.map((suggestion) => ({
    suggestion,
    proof: modelProof(installed.find((entry) => entry.repo === suggestion.repo), preflights[suggestion.repo]?.repo === suggestion.repo ? preflights[suggestion.repo] : null, status.memoryLimitBytes, suggestion.repo === QUALIFIED_LARGE),
  })).sort((a, b) => rank(a.suggestion.repo) - rank(b.suggestion.repo))
  const ready = candidates.flatMap(({ suggestion, proof }): RecommendedEngineModel[] => proof.state === 'ready' ? [{
    repo: suggestion.repo, title: suggestion.title, installed: proof.installed, action: proof.installed ? 'load' : 'download', memoryBytes: proof.memoryBytes,
    reason: modelReason(suggestion.repo, proof, qualifiedChip, flashChip),
  }] : [])
  if (!ready.length) return candidates.some((candidate) => candidate.proof.state === 'checking')
    ? result('checking', 'Checking model compatibility, memory, and download space before recommending one.')
    : result('unavailable', 'None of the curated models have passed the checks for this device; inspect the model checks or try again.')
  const selected = ready[0]
  const fallback = ready.slice(1).filter((candidate) => candidate.memoryBytes < selected.memoryBytes).sort((a, b) => a.memoryBytes - b.memoryBytes)[0] ?? null
  return { status: 'ready', selected, fallback, reason: selected.reason, hardwareLabel: label }
}
