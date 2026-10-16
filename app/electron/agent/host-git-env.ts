import { execFile } from 'node:child_process'

// Git commands the app runs on the host (status, diff, snapshots, commits) run outside the
// agent sandbox, so a repository's own config must not be able to execute code through
// them. GIT_CONFIG_COUNT/KEY/VALUE act like `-c` flags (git 2.31+) and win over every
// config file. Hooks stay off as well: host commits already skip them on purpose.
const HOST_GIT_OVERRIDES: Array<[string, string]> = [
  ['core.fsmonitor', 'false'],
  ['core.hooksPath', '/dev/null'],
]

/** `base` plus the host git overrides, appended after any GIT_CONFIG_* entries it already carries. */
export function hostGitEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...base }
  const existing = Number.parseInt(env.GIT_CONFIG_COUNT ?? '', 10)
  let count = Number.isSafeInteger(existing) && existing > 0 ? existing : 0
  for (const [key, value] of HOST_GIT_OVERRIDES) {
    env[`GIT_CONFIG_KEY_${count}`] = key
    env[`GIT_CONFIG_VALUE_${count}`] = value
    count++
  }
  env.GIT_CONFIG_COUNT = String(count)
  return env
}

/** Disable repository content/merge helpers before host Git stages or restores files. */
export async function hostGitEnvForWorkspace(cwd?: string, base: NodeJS.ProcessEnv = process.env): Promise<NodeJS.ProcessEnv> {
  const env = hostGitEnv(base)
  const keys = await new Promise<string>((resolve, reject) => {
    execFile('git', ['config', '--null', '--name-only', '--get-regexp', '^(filter\\..*\\.(clean|smudge|process|required)|merge\\..*\\.driver)$'], {
      cwd, env, encoding: 'utf8', timeout: 5_000, maxBuffer: 256_000,
    }, (error, stdout) => {
      // Exit 1 means no matching settings. A failed or truncated inspection is
      // not permission to run a host command with unreviewed filters.
      if (error && error.code !== 1) reject(new Error('Cannot inspect Git content filters safely.'))
      else resolve(stdout)
    })
  })
  let count = Number(env.GIT_CONFIG_COUNT)
  for (const key of new Set(keys.split('\0').filter(Boolean))) {
    if (!/^(filter\..+\.(clean|smudge|process|required)|merge\..+\.driver)$/.test(key)) throw new Error('Invalid Git helper setting.')
    env[`GIT_CONFIG_KEY_${count}`] = key
    env[`GIT_CONFIG_VALUE_${count}`] = key.startsWith('merge.') || key.endsWith('.required') ? 'false' : ''
    count++
  }
  // A repository-selected signing program is another executable helper.
  env[`GIT_CONFIG_KEY_${count}`] = 'commit.gpgSign'
  env[`GIT_CONFIG_VALUE_${count++}`] = 'false'
  env.GIT_CONFIG_COUNT = String(count)
  return env
}
