import type { EngineEvent, EngineInstalledModel, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Readiness, SuggestedModel } from '../types'
import type { TensorFoldStudioApi } from '../ipc'

// Browser demo of the built-in engine: a fake installed model, Hub results and a simulated download.
// Nothing here reaches Hugging Face or runs a model.

const GIB = 1024 ** 3
const TOTAL_MEMORY = 36 * GIB
const LIMIT = Math.floor(TOTAL_MEMORY * 0.7)

const SUGGESTIONS: SuggestedModel[] = [
  { repo: 'mlx-community/Qwen3.5-2B-MLX-4bit', revision: '93760be4f1f69842a46bc13dbdc0f19e291392a3', title: 'Qwen3.5-2B', note: 'The pinned 4-bit checkpoint. Small enough for most Macs.' },
  { repo: 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit', revision: null, title: 'Nemotron 3.5 Lightning 30B-A3B', note: 'Qualified on Metal from M1 to M5.' },
  { repo: 'TensorFold/Qwen3.8-27B-MLX-4bit', revision: null, title: 'Qwen3.8-27B', note: 'Downloads with its DFlash2 drafter. Qualified on M5 Max and M3 Ultra.' },
  { repo: 'TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP', revision: null, title: 'Qwen3.8 Flash Next', note: 'The 6-bit checkpoint, about 147 GiB, with built-in multi-token drafting. Needs a Mac with 256 GB of memory; qualified on M5 Ultra.' },
]

const SIZES: Record<string, number> = {
  'mlx-community/Qwen3.5-2B-MLX-4bit': 1.6 * GIB,
  'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit': 17.8 * GIB,
  'TensorFold/Qwen3.8-27B-MLX-4bit': 16.1 * GIB,
  'TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP': 147.2 * GIB,
}

const HUB: HubModelSummary[] = [
  { repo: 'mlx-community/Qwen3.5-2B-MLX-4bit', downloads: 48210, likes: 51, lastModified: '2026-08-30T10:00:00.000Z', gated: false, modelType: 'qwen3_5', pipelineTag: 'text-generation', libraryName: 'mlx' },
  { repo: 'mlx-community/Qwen3.5-2B-8bit', downloads: 9120, likes: 8, lastModified: '2026-08-30T10:00:00.000Z', gated: false, modelType: 'qwen3_5', pipelineTag: 'text-generation', libraryName: 'mlx' },
  { repo: 'TensorFold/Qwen3.8-27B-MLX-4bit', downloads: 20450, likes: 210, lastModified: '2026-10-01T10:00:00.000Z', gated: false, modelType: 'qwen3_5', pipelineTag: 'text-generation', libraryName: 'mlx' },
  { repo: 'google/gemma-4-12b-it', downloads: 310200, likes: 1200, lastModified: '2026-07-01T10:00:00.000Z', gated: 'manual', modelType: 'gemma4', pipelineTag: 'text-generation', libraryName: 'transformers' },
]

export function createMockEngineApi(): Pick<TensorFoldStudioApi,
  'engineSnapshot' | 'engineConnection' | 'engineRefresh' | 'engineSearch' | 'enginePreflight' | 'enginePull' | 'engineCancelPull' | 'engineLoad' | 'engineUnload' | 'engineSetCortheon' | 'engineDelete' | 'onEngineEvent'> & { mockEngineReadiness(model: string | null): Readiness } {
  const listeners = new Set<(event: EngineEvent) => void>()
  const emit = (event: EngineEvent) => { for (const cb of listeners) cb(event) }
  let installed: EngineInstalledModel[] = [{
    repo: 'mlx-community/Qwen3.5-2B-MLX-4bit', family: 'Qwen3.5-2B or Qwen3.8-27B', modelType: 'qwen3_5', format: '4-bit, group 64', contextTokens: 262144,
    weightsBytes: 1.6 * GIB, memoryFloorBytes: 3.6 * GIB, fitsMemory: true, drafter: null, drafterInstalled: null,
  }]
  const pulls = new Map<string, PullProgress>()
  const timers = new Map<string, ReturnType<typeof setInterval>>()
  let status: EngineStatus = {
    hardware: { platform: 'darwin', architecture: 'arm64', chip: 'Apple M3 Max' },
    available: true, version: '1.0.9', binaryPath: '/Applications/TensorFold.app/Contents/Resources/engine/bin/tensorfold-native',
    state: 'stopped', model: null, drafter: null, port: null, startedAt: null, readyAt: null, message: null, logTail: [], totalMemoryBytes: TOTAL_MEMORY, memoryLimitBytes: LIMIT, contextTokens: null,
    cortheon: { supported: true, enabled: false, active: false },
  }
  const setStatus = (patch: Partial<EngineStatus>) => { status = { ...status, ...patch }; emit({ kind: 'status', status }) }
  const snapshot = (): EngineSnapshot => ({ status, performance: null, installed, pulls: [...pulls.values()], suggestions: SUGGESTIONS, hubCache: '~/.cache/huggingface/hub' })

  const preflight = async (repo: string, revision?: string | null): Promise<ModelPreflight> => {
    await new Promise((r) => setTimeout(r, 250))
    const hit = HUB.find((h) => h.repo === repo)
    const modelType = hit?.modelType ?? (repo.includes('Nemotron') ? 'nemotron_h' : 'qwen3_5')
    const supported = modelType === 'qwen3_5' || modelType === 'nemotron_h'
    const format = repo.includes('8bit') ? 'mlx-q8g64' : 'mlx-q4g64'
    const total = SIZES[repo] ?? 8 * GIB
    const floor = total + 2 * GIB
    const gated = hit?.gated ?? false
    const isInstalled = installed.some((m) => m.repo === repo)
    const reasons: string[] = []
    let verdict: ModelPreflight['verdict'] = 'ok'
    if (!supported) { verdict = 'unsupported'; reasons.push(`The built-in engine serves no ${modelType} family yet.`) }
    else if (format !== 'mlx-q4g64') { verdict = 'format'; reasons.push('The engine reads 4-bit, group 64 for this family; this checkpoint is 8-bit, group 64.') }
    else if (gated) { verdict = 'gated'; reasons.push('The repo is gated, and the engine downloads without a Hugging Face token.') }
    else if (floor > LIMIT) { verdict = 'too-big'; reasons.push('It needs more memory than this Mac allows.') }
    else if (isInstalled) verdict = 'installed'
    return {
      repo, revision: revision ?? null, verdict, reasons, modelType, family: modelType === 'nemotron_h' ? 'Nemotron 3.5 Lightning' : supported ? 'Qwen3.5-2B or Qwen3.8-27B' : null, supported, format, formatOk: supported ? format === 'mlx-q4g64' : null, gated,
      totalBytes: total, fileCount: 6, drafter: repo === 'TensorFold/Qwen3.8-27B-MLX-4bit' ? { repo: 'z-lab/Qwen3.8-27B-DFlash2', totalBytes: 1.2 * GIB } : null,
      memoryFloorBytes: floor, memoryLimitBytes: LIMIT, fitsMemory: floor <= LIMIT, freeDiskBytes: 412 * GIB, neededDiskBytes: isInstalled ? 0 : total, fitsDisk: true, cachedBytes: isInstalled ? total : 0, installed: isInstalled, checkedAt: Date.now(),
    }
  }

  return {
    engineSnapshot: async () => snapshot(),
    // Demo mode never starts a server, so it cannot supply a working client connection.
    engineConnection: async () => null,
    engineRefresh: async () => snapshot(),
    engineSearch: async (query) => {
      await new Promise((r) => setTimeout(r, 300))
      const q = query.trim().toLowerCase()
      return q ? HUB.filter((h) => h.repo.toLowerCase().includes(q) || q.split(/\s+/).some((word) => h.repo.toLowerCase().includes(word))) : []
    },
    enginePreflight: preflight,
    enginePull: async (repo, revision) => {
      const total = SIZES[repo] ?? 8 * GIB
      const progress: PullProgress = { repo, revision: revision ?? null, state: 'running', phase: `Downloading ${repo}`, receivedBytes: 0, totalBytes: total, lastLine: `[tensorfold] downloading ${repo}@main from Hugging Face`, message: null, startedAt: Date.now(), finishedAt: null }
      pulls.set(repo, progress)
      emit({ kind: 'pull', pull: progress })
      timers.set(repo, setInterval(() => {
        const current = pulls.get(repo)
        if (!current || current.state !== 'running') return
        const receivedBytes = Math.min(total, current.receivedBytes + total / 12)
        const done = receivedBytes >= total
        const next: PullProgress = { ...current, receivedBytes, state: done ? 'done' : 'running', phase: done ? 'Downloaded' : current.phase, lastLine: done ? `stored ${repo}@93760be at ~/.cache/huggingface/hub` : `  model-0000${Math.ceil((receivedBytes / total) * 4)}-of-00004.safetensors: 2048.00 MiB in 64 pieces`, finishedAt: done ? Date.now() : null }
        pulls.set(repo, next)
        emit({ kind: 'pull', pull: next })
        if (done) {
          clearInterval(timers.get(repo))
          const nemotron = repo.includes('Nemotron')
          installed = [...installed.filter((m) => m.repo !== repo), { repo, family: nemotron ? 'Nemotron 3.5 Lightning' : 'Qwen3.5-2B or Qwen3.8-27B', modelType: nemotron ? 'nemotron_h' : 'qwen3_5', format: '4-bit, group 64', contextTokens: 262144, weightsBytes: total, memoryFloorBytes: total + 2 * GIB, fitsMemory: total + 2 * GIB <= LIMIT, drafter: null, drafterInstalled: null }]
          emit({ kind: 'installed', installed })
        }
      }, 500))
      return progress
    },
    engineCancelPull: async (repo) => {
      const current = pulls.get(repo)
      clearInterval(timers.get(repo))
      if (!current || current.state !== 'running') return { cancelled: false }
      const next: PullProgress = { ...current, state: 'cancelled', phase: 'Cancelled', message: 'Download cancelled. Download again to resume.', finishedAt: Date.now() }
      pulls.set(repo, next)
      emit({ kind: 'pull', pull: next })
      return { cancelled: true }
    },
    engineLoad: async (repo) => {
      setStatus({ state: 'starting', model: repo, startedAt: Date.now(), readyAt: null, message: `Loading ${repo}`, port: 52814 })
      await new Promise((r) => setTimeout(r, 1200))
      setStatus({ state: 'ready', readyAt: Date.now(), message: null, contextTokens: 32_768, cortheon: { supported: true, enabled: status.cortheon?.enabled ?? false, active: status.cortheon?.enabled ?? false } })
      return status
    },
    engineUnload: async () => {
      setStatus({ state: 'stopped', model: null, port: null, startedAt: null, readyAt: null, message: null, contextTokens: null, cortheon: { supported: true, enabled: status.cortheon?.enabled ?? false, active: false } })
      return status
    },
    engineSetCortheon: async (enabled) => {
      const running = status.state === 'ready'
      setStatus({ cortheon: { supported: true, enabled, active: running && enabled } })
      return status
    },
    engineDelete: async (repo) => {
      if (status.model === repo && status.state !== 'stopped') throw new Error(`Unload ${repo} before deleting it.`)
      installed = installed.filter((m) => m.repo !== repo)
      emit({ kind: 'installed', installed })
      return installed
    },
    onEngineEvent: (cb) => { listeners.add(cb); return () => { listeners.delete(cb) } },
    mockEngineReadiness: (model) => {
      if (!installed.length) return { ready: false, reason: 'needs-model', message: 'Download a model for the built-in TensorFold engine in Models.' }
      if (model && !installed.some((m) => m.repo === model)) return { ready: false, reason: 'needs-model', message: `${model} is not downloaded. Open Models to get it.` }
      return { ready: true, reason: 'ok', message: '' }
    },
  }
}
