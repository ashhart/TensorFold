import os from 'node:os'
import { resolveHubEndpoint } from './hub-network'

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

// Snapshot before filtering so Hub checks and native pulls cannot select different origins.
export function engineLaunchEnv(options: { childEnv?: NodeJS.ProcessEnv; env?: NodeJS.ProcessEnv }) {
  const source = { ...(options.childEnv ?? options.env ?? process.env) }
  const endpoint = resolveHubEndpoint(source.HF_ENDPOINT)
  const child = options.childEnv ? source : engineChildEnv(source)
  delete child.HF_TOKEN
  child.HF_ENDPOINT = endpoint
  return { env: Object.freeze(child), endpoint }
}
