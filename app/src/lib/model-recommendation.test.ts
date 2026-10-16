import assert from 'node:assert/strict'
import type { EngineInstalledModel, EngineStatus, ModelPreflight, SuggestedModel } from './types'
import { recommendEngineModel, type ModelRecommendationInput } from './model-recommendation'

const GIB = 1024 ** 3
const LIGHT = 'mlx-community/Qwen3.5-2B-MLX-4bit'
const BALANCED = 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
const LARGE = 'TensorFold/Qwen3.8-27B-MLX-4bit'
const FLASH = 'TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP'
const suggestions: SuggestedModel[] = [
  { repo: LIGHT, title: 'Qwen3.5-2B', revision: null, note: 'Lightweight' },
  { repo: BALANCED, title: 'Nemotron 3.5 Lightning 30B-A3B', revision: null, note: 'Qualified from M1 to M5' },
  { repo: LARGE, title: 'Qwen3.8-27B', revision: null, note: 'Qualified on M5 Max and M3 Ultra' },
]
const status = (chip = 'Apple M3 Max', memoryGiB = 36): EngineStatus => ({
  available: true, version: '1.0.8', binaryPath: '/engine/bin/tensorfold-native', state: 'stopped', model: null, drafter: null, port: null,
  startedAt: null, readyAt: null, message: null, logTail: [], totalMemoryBytes: memoryGiB * GIB, memoryLimitBytes: Math.floor(memoryGiB * GIB * 0.7), contextTokens: null,
  hardware: { platform: 'darwin', architecture: 'arm64', chip },
})
function preflight(repo: string, memoryGiB: number, patch: Partial<ModelPreflight> = {}): ModelPreflight {
  return {
    repo, revision: null, verdict: 'ok', reasons: [], modelType: repo === BALANCED ? 'nemotron_h' : 'qwen3_5', family: 'Example family',
    supported: true, format: 'mlx-q4g64', formatOk: true, gated: false, totalBytes: (memoryGiB - 2) * GIB, fileCount: 6,
    drafter: repo === LARGE ? { repo: 'z-lab/Qwen3.8-27B-DFlash2', totalBytes: GIB } : null,
    memoryFloorBytes: memoryGiB * GIB, memoryLimitBytes: 25.2 * GIB, fitsMemory: true, freeDiskBytes: 100 * GIB,
    neededDiskBytes: (memoryGiB - 2) * GIB, fitsDisk: true, cachedBytes: 0, installed: false, checkedAt: 1, ...patch,
  }
}
const validChecks = () => ({ [LIGHT]: preflight(LIGHT, 4), [BALANCED]: preflight(BALANCED, 20), [LARGE]: preflight(LARGE, 19) })
const input = (patch: Partial<ModelRecommendationInput> = {}): ModelRecommendationInput => ({ status: status(), suggestions, installed: [], preflights: validChecks(), ...patch })
const installed = (repo: string, memoryGiB: number, patch: Partial<EngineInstalledModel> = {}): EngineInstalledModel => ({
  repo, family: 'Example family', modelType: 'qwen3_5', format: '4-bit, group 64', contextTokens: 32768, weightsBytes: (memoryGiB - 2) * GIB,
  memoryFloorBytes: memoryGiB * GIB, fitsMemory: true, drafter: repo === LARGE ? 'z-lab/Qwen3.8-27B-DFlash2' : null, drafterInstalled: repo === LARGE ? true : null, ...patch,
})

{
  const recommendation = recommendEngineModel(input())
  assert.equal(recommendation.status, 'ready')
  assert.equal(recommendation.selected?.repo, BALANCED, 'Nemotron is the balanced workhorse when it passes the device checks')
  assert.equal(recommendation.selected?.action, 'download')
  assert.equal(recommendation.selected?.installed, false)
  assert.equal(recommendation.fallback?.repo, LIGHT)
  assert.match(recommendation.reason, /balanced model/)
  assert.equal(recommendation.hardwareLabel, 'Apple M3 Max with 36 GiB memory')
}

{
  const small = recommendEngineModel(input({ status: status('Apple M1', 8) }))
  assert.equal(small.selected?.repo, LIGHT, 'Current memory limits override a preflight from a larger Mac')
  assert.equal(small.fallback, null)
  for (const chip of ['Apple M5 Max', 'Apple M3 Ultra', 'apple m5 max']) {
    const recommendation = recommendEngineModel(input({ status: status(chip, 64) }))
    assert.equal(recommendation.selected?.repo, LARGE)
    assert.match(recommendation.selected.reason, /qualified on this chip/)
  }
  for (const chip of ['Apple M5', 'Apple M5 Pro', 'Apple M3 Max', 'Apple M4 Max']) {
    assert.equal(recommendEngineModel(input({ status: status(chip, 64) })).selected?.repo, BALANCED, 'Large Qwen is elevated only on its qualified chips')
  }
  const onlyLarge = recommendEngineModel(input({ suggestions: [suggestions[2]], status: status('Apple M4 Max', 64) }))
  assert.equal(onlyLarge.status, 'ready', 'A different chip is not invented as an unsupported model family')
  assert.match(onlyLarge.reason, /qualified on M5 Max and M3 Ultra/)
  assert.doesNotMatch(onlyLarge.reason, /unsupported|cannot run/i)
}

{
  const recommendation = recommendEngineModel(input({ installed: [installed(BALANCED, 20)], preflights: {} }))
  assert.equal(recommendation.selected?.repo, BALANCED, 'An installed, memory-checked model needs no network preflight')
  assert.equal(recommendation.selected?.action, 'load')
  assert.equal(recommendation.selected?.installed, true)
  assert.equal(recommendation.fallback, null, 'Unknown compatibility must not become a fallback')
  const tooLarge = recommendEngineModel(input({ status: status('Apple M1', 8), installed: [installed(BALANCED, 20)], preflights: {} }))
  assert.notEqual(tooLarge.status, 'ready', 'Installed metadata must fit this device, not only its cached fitsMemory flag')
  assert.equal(recommendEngineModel(input({ suggestions: [suggestions[1]], status: status('Apple M1', 8), installed: [installed(BALANCED, 20)], preflights: {} })).status, 'unavailable', 'A known memory failure is not left checking forever')
}

{
  const onlyLarge = [suggestions[2]]
  assert.equal(recommendEngineModel(input({ suggestions: onlyLarge, installed: [installed(LARGE, 19, { drafterInstalled: false })], preflights: {} })).status, 'checking')
  assert.equal(recommendEngineModel(input({ suggestions: onlyLarge, installed: [installed(LARGE, 19, { drafter: null })], preflights: {} })).status, 'checking')
  const missing = recommendEngineModel(input({ suggestions: onlyLarge, installed: [installed(LARGE, 19, { drafterInstalled: false })] }))
  assert.equal(missing.status, 'ready', 'A complete preflight can recommend downloading the missing companion')
  assert.equal(missing.selected?.action, 'download', 'A cached main model without its drafter is not ready to load')
  for (const drafter of [null, { repo: 'z-lab/Qwen3.8-27B-DFlash2', totalBytes: null }]) {
    assert.equal(recommendEngineModel(input({ suggestions: onlyLarge, preflights: { [LARGE]: preflight(LARGE, 19, { drafter }) } })).status, 'checking')
  }
}

{
  assert.equal(recommendEngineModel(input({ preflights: {} })).status, 'checking')
  assert.equal(recommendEngineModel(input({ preflights: { [LIGHT]: null } })).selected, null)
  for (const patch of [
    { formatOk: null }, { format: null }, { fitsMemory: null }, { fitsDisk: null }, { memoryFloorBytes: null }, { totalBytes: null },
    { freeDiskBytes: null }, { neededDiskBytes: null }, { memoryLimitBytes: Number.NaN }, { totalBytes: Number.POSITIVE_INFINITY },
  ] as Partial<ModelPreflight>[]) {
    const recommendation = recommendEngineModel(input({ suggestions: [suggestions[0]], preflights: { [LIGHT]: preflight(LIGHT, 4, patch) } }))
    assert.equal(recommendation.status, 'checking', `Unknown numeric/compatibility evidence cannot claim ready: ${JSON.stringify(patch)}`)
    assert.equal(recommendation.selected, null)
    assert.equal(recommendation.fallback, null)
  }
  for (const patch of [
    { supported: false }, { formatOk: false }, { fitsMemory: false }, { fitsDisk: false }, { gated: 'manual' }, { freeDiskBytes: 1 },
    { verdict: 'unsupported' }, { verdict: 'format' }, { verdict: 'gated' }, { verdict: 'too-big' }, { verdict: 'disk' }, { verdict: 'error' },
  ] as Partial<ModelPreflight>[]) {
    assert.equal(recommendEngineModel(input({ suggestions: [suggestions[0]], preflights: { [LIGHT]: preflight(LIGHT, 4, patch) } })).status, 'unavailable')
  }
  assert.equal(recommendEngineModel(input({ suggestions: [suggestions[0]], preflights: { [LIGHT]: preflight('different/repo', 4) } })).status, 'checking', 'Evidence for another repository cannot prove this one')
  assert.equal(recommendEngineModel(input({ suggestions: [], installed: [installed(BALANCED, 20)] })).status, 'unavailable', 'Installed repos outside the current curated suggestions are not silently selected')
}

{
  for (const hardware of [
    { platform: 'darwin', architecture: 'x64', chip: 'Intel CPU' },
    { platform: 'linux', architecture: 'arm64', chip: 'ARM CPU' },
    { platform: 'win32', architecture: 'x64', chip: 'AMD CPU' },
  ]) {
    const recommendation = recommendEngineModel(input({ status: { ...status(), hardware } }))
    assert.equal(recommendation.status, 'unsupported')
    assert.equal(recommendation.selected, null)
    assert.equal(recommendation.fallback, null)
    assert.match(recommendation.reason, /macOS on Apple silicon/)
  }
  assert.equal(recommendEngineModel(input({ status: null })).status, 'checking')
  assert.equal(recommendEngineModel(input({ status: { ...status(), hardware: undefined } })).status, 'checking')
  assert.equal(recommendEngineModel(input({ status: { ...status(), available: false } })).status, 'unavailable')
  assert.equal(recommendEngineModel(input({ status: { ...status(), memoryLimitBytes: 0 } })).status, 'checking')
  assert.equal(recommendEngineModel(input({ status: { ...status(), hardware: { platform: 'darwin', architecture: 'arm64', chip: null } } })).selected?.repo, BALANCED)
}

console.log('Hardware-aware model recommendations require complete compatibility evidence, respect memory/drafter limits, rank qualified chips, and reject unsupported builds.')

{
  // Flash Next joins only when its 147 GiB floor fits; it leads on M5 Ultra, its qualified chip.
  const withFlash = [...suggestions, { repo: FLASH, title: 'Qwen3.8 Flash Next', revision: null, note: 'Qualified on M5 Ultra' }]
  const checks = (limitGiB: number) => ({ ...validChecks(), [FLASH]: preflight(FLASH, 150, { format: 'mlx-q6g32', modelType: 'qwen4_exp', memoryLimitBytes: limitGiB * GIB, fitsMemory: 150 <= limitGiB, freeDiskBytes: 400 * GIB }) })
  const ultra = recommendEngineModel(input({ suggestions: withFlash, status: status('Apple M5 Ultra', 256), preflights: checks(179) }))
  assert.equal(ultra.selected?.repo, FLASH, 'Flash Next leads on an M5 Ultra that can hold it')
  assert.match(ultra.selected.reason, /largest model this Mac can hold/)
  assert.ok(ultra.fallback && ultra.fallback.memoryBytes < ultra.selected.memoryBytes, 'a smaller fallback is offered')
  const laptop = recommendEngineModel(input({ suggestions: withFlash, status: status('Apple M5 Max', 128), preflights: checks(89.6) }))
  assert.equal(laptop.selected?.repo, LARGE, 'a 128 GB M5 Max cannot hold Flash Next, so the 27B leads')
  const m3 = recommendEngineModel(input({ suggestions: withFlash, status: status('Apple M3 Ultra', 512), preflights: checks(358) }))
  assert.equal(m3.selected?.repo, LARGE, 'off its qualified chip Flash Next stays available but does not lead')
  const onlyFlash = recommendEngineModel(input({ suggestions: [withFlash[3]], status: status('Apple M3 Ultra', 512), preflights: checks(358) }))
  assert.match(onlyFlash.reason, /qualified on M5 Ultra/)
}

console.log('model recommendation: Flash Next ranking ok')
