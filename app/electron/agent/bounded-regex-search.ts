import { spawn } from 'node:child_process'

const GREP_PATH = process.platform === 'win32' ? 'grep' : '/usr/bin/grep'
const SEARCH_TIMEOUT_MS = 3_000
const SEARCH_OUTPUT_BYTES = 2_000_000
const SEARCH_QUERY_CHARS = 2_000

/**
 * Run model-supplied regex evaluation out of Electron's main thread. A pathological
 * expression can consume only the bounded grep child and is force-killed on timeout
 * or output overflow, so JavaScript timers/IPC remain responsive.
 */
export function boundedRegexSearch(
  cwd: string,
  query: string,
  skipDirectories: Iterable<string>,
): Promise<string[]> {
  if (!query) return Promise.resolve([])
  if (query.length > SEARCH_QUERY_CHARS) return Promise.reject(new Error(`Search pattern exceeds ${SEARCH_QUERY_CHARS} characters.`))
  const args = ['-rIinE', ...[...skipDirectories].map((name) => `--exclude-dir=${name}`), '--', query, '.']
  return new Promise((resolve, reject) => {
    const child = spawn(GREP_PATH, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    let bytes = 0
    let settled = false
    let timedOut = false

    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      fn()
    }
    const stop = () => {
      try { child.kill('SIGKILL') } catch { /* already exited */ }
    }
    const timer = setTimeout(() => {
      timedOut = true
      stop()
    }, SEARCH_TIMEOUT_MS)
    timer.unref?.()

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      bytes += Buffer.byteLength(chunk)
      if (bytes > SEARCH_OUTPUT_BYTES) { stop(); return }
      stdout += chunk
    })
    child.stderr.on('data', (chunk: string) => {
      if (stderr.length < 4_000) stderr += chunk.slice(0, 4_000 - stderr.length)
    })
    child.on('error', (error) => finish(() => reject(new Error(`Search process failed: ${error.message}`))))
    child.on('close', (code) => finish(() => {
      if (timedOut) return reject(new Error(`Search regex exceeded the ${SEARCH_TIMEOUT_MS}ms safety limit.`))
      if (bytes > SEARCH_OUTPUT_BYTES) return reject(new Error('Search output exceeded the safety limit. Narrow the query or path.'))
      // grep: 0 matches, 1 no matches, 2 syntax/runtime error.
      if (code !== 0 && code !== 1) return reject(new Error(`Invalid or failed search regex${stderr.trim() ? `: ${stderr.trim().slice(0, 240)}` : '.'}`))
      resolve(stdout.split('\n').filter(Boolean))
    }))
  })
}
