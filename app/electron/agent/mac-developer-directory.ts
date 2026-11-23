import { realpathSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'

interface DeveloperDirectoryIO {
  realpath(value: string): string
  selectedPath(): string | null
}

const systemIO: DeveloperDirectoryIO = {
  realpath: realpathSync,
  selectedPath() {
    const result = spawnSync('/usr/bin/xcode-select', ['--print-path'], {
      encoding: 'utf8', timeout: 2000, env: { PATH: '/usr/bin:/bin' },
    })
    return result.status === 0 ? result.stdout.trim() : null
  },
}

export function resolveDeveloperReadRoot(io: DeveloperDirectoryIO = systemIO): string | null {
  const resolve = (value: string | null): string | null => {
    if (!value || !path.isAbsolute(value)) return null
    try { return io.realpath(value) } catch { return null }
  }
  let directory = resolve('/var/db/xcode_select_link')
  if (!directory) {
    try { directory = resolve(io.selectedPath()) } catch { /* retain the Command Line Tools fallback */ }
  }
  directory ??= resolve('/Library/Developer/CommandLineTools')
  // Xcode shims also load SharedFrameworks outside Contents/Developer.
  return directory?.match(/^(.*?\.app)\/Contents\/Developer$/)?.[1] ?? directory
}
