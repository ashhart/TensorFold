import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EngineStatus } from '../../types'
import type { InstalledModels } from '../installed-models'
import { isLocalModelId } from '../library'
import type { ModelRevisions } from '../model-revisions'
import { hasCachedSnapshot, repoCacheDir } from '../paths'
import { assertBundledBenchmarkVersion } from './version'

// A benchmark runs only against a cached public checkpoint the app can pin by revision and config hash.

export interface BenchmarkModelCheckDeps {
  installed: InstalledModels
  revisions: ModelRevisions
  hub: () => string
  status: () => Promise<Pick<EngineStatus, 'version'>>
}

export async function checkBenchmarkModel(deps: BenchmarkModelCheckDeps, model: string): Promise<{ revision: string; configSha256: string; engineVersion: string }> {
  const installed = await deps.installed.list(true)
  const entry = installed.find(item => item.repo === model)
  if (!entry || entry.fitsMemory !== true || entry.source === 'library' || isLocalModelId(model) || !(await hasCachedSnapshot(deps.hub(), model))) throw new Error('Choose a cached installed checkpoint confirmed to fit this Mac.')
  const revision = (await fs.readFile(path.join(repoCacheDir(deps.hub(), model), 'refs', 'main'), 'utf8')).trim()
  const snapshot = await deps.revisions.snapshot(model, revision)
  const argument = await deps.installed.argument(model)
  if (argument !== model && await fs.realpath(argument) !== snapshot) throw new Error('The selected checkpoint differs from the cached public model.')
  const status = await deps.status()
  const engineVersion = assertBundledBenchmarkVersion(status.version)
  await deps.installed.info(model)
  return { revision, configSha256: createHash('sha256').update(await fs.readFile(path.join(snapshot, 'config.json'))).digest('hex'), engineVersion }
}
