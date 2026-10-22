import os from 'node:os'
import { resolveHubEndpoint } from './hub-network'
import { PINNED_ENGINE_ENV_KEYS } from '../../../shared/engine-advanced-env'

const CHILD_ENV_KEYS = ['HOME', 'PATH', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'USER', 'LOGNAME', 'SHELL', 'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'HF_ENDPOINT']

// The engine gets ordinary OS values and the Hugging Face cache settings, never the app's provider keys.
export function engineChildEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const key of CHILD_ENV_KEYS) if (typeof base[key] === 'string') env[key] = base[key]
  if (env.HF_ENDPOINT) env.HF_ENDPOINT = resolveHubEndpoint(env.HF_ENDPOINT)
  env.HOME = env.HOME || os.homedir()
  env.PATH = env.PATH || '/usr/bin:/bin:/usr/sbin:/sbin'
  return env
}

// A child's view of `base`: the Hub origin that authorized its checks, and no Hub token.
// Every pinned key is dropped first, so a value inherited from the app's own environment
// cannot reach the engine either; the advanced page refuses these names for the same reason.
export function pinnedEngineEnv(base: NodeJS.ProcessEnv, endpoint: string): NodeJS.ProcessEnv {
  const child = { ...base }
  delete child.HF_TOKEN
  for (const key of PINNED_ENGINE_ENV_KEYS) delete child[key]
  child.HF_ENDPOINT = endpoint
  return Object.freeze(child)
}

// Snapshot before filtering so Hub checks and native pulls cannot select different origins.
export function engineLaunchEnv(options: { childEnv?: NodeJS.ProcessEnv; env?: NodeJS.ProcessEnv }) {
  const source = { ...(options.childEnv ?? options.env ?? process.env) }
  const endpoint = resolveHubEndpoint(source.HF_ENDPOINT)
  return { env: pinnedEngineEnv(options.childEnv ? source : engineChildEnv(source), endpoint), endpoint }
}
