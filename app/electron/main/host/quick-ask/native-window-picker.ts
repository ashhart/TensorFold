import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { lstat } from 'node:fs/promises'
import os from 'node:os'
import { APP_BUNDLE_ID } from '../app-identity'
import { capturedWindowAttachments } from '../../../agent/quick-ask/window-context'

export function supportsNativeAskPicker(platform = process.platform, release = os.release()): boolean {
  const [major, minor] = release.split('.').map(Number)
  return platform === 'darwin' && (major > 24 || major === 24 && minor >= 2)
}

export function createNativeAskPicker(binary: () => string, options: { launch?: typeof spawn; stat?: typeof lstat; timeoutMs?: number } = {}) {
  let child: ChildProcessWithoutNullStreams | null = null, running = false, cancelled = false
  let phase: 'starting' | 'picking' | 'capturing' = 'starting'
  return {
    get busy() { return running },
    cancel() {
      if (child && phase !== 'capturing') return false
      cancelled = true; child?.kill('SIGTERM'); return true
    },
    async pick() {
      if (running) throw new Error('Finish or cancel the current window selection first.')
      running = true; cancelled = false; phase = 'starting'
      try {
        const file = binary(), info = await (options.stat ?? lstat)(file)
        if (!info.isFile() || info.isSymbolicLink() || !(info.mode & 0o111)) throw new Error('The window picker helper is missing. Reinstall TensorFold.')
        if (cancelled) return null
        const raw = await new Promise<unknown>((resolve, reject) => {
          const process = (options.launch ?? spawn)(file, ['--ask-window-picker', '--own-bundle-id', APP_BUNDLE_ID], {
            env: { PATH: '/usr/bin:/bin:/usr/sbin:/sbin', LANG: 'en_US.UTF-8', HOME: os.homedir() }, stdio: ['pipe', 'pipe', 'pipe'],
          })
          child = process
          let pending = Buffer.alloc(0), length = 0, settled = false, hasResult = false, result: unknown
          const finish = (error?: Error, value?: unknown) => {
            if (settled) return
            settled = true; clearTimeout(timer)
            if (error) { process.kill('SIGKILL'); reject(error) } else resolve(value)
          }
          const timedOut = () => finish(new Error('The window picker stopped responding. Try again.'))
          let timer = setTimeout(timedOut, options.timeoutMs ?? 75_000)
          timer.unref()
          const consume = (line: Buffer) => {
            if (!line.length) return
            const value = JSON.parse(line.toString('utf8'))
            if (value && typeof value === 'object' && 'state' in value) {
              if (hasResult || Object.keys(value).join(',') !== 'state'
                || !(phase === 'starting' && value.state === 'picking' || phase === 'picking' && value.state === 'capturing')) {
                throw new Error('The window picker returned an invalid state.')
              }
              phase = value.state; clearTimeout(timer)
              if (phase === 'capturing') { timer = setTimeout(timedOut, options.timeoutMs ?? 45_000); timer.unref() }
              return
            }
            if (hasResult) throw new Error('The window picker returned duplicate context.')
            clearTimeout(timer)
            timer = setTimeout(timedOut, options.timeoutMs ?? 5_000); timer.unref()
            hasResult = true; result = value
          }
          process.stdout.on('data', (bytes: Buffer) => {
            length += bytes.length
            if (length > 2_700_000) { finish(new Error('The selected window exceeds the context limit.')); return }
            pending = Buffer.concat([pending, bytes])
            try {
              let end: number
              while ((end = pending.indexOf(10)) >= 0) {
                consume(pending.subarray(0, end)); pending = pending.subarray(end + 1)
              }
            } catch { finish(new Error('The window picker returned invalid context.')) }
          })
          process.stderr.on('data', () => {})
          process.on('error', () => finish(new Error('The window picker could not start. Reinstall TensorFold.')))
          process.on('close', () => {
            if (cancelled) { finish(undefined, { cancelled: true }); return }
            try { consume(pending); if (!hasResult) throw new Error('Missing result'); finish(undefined, result) }
            catch { finish(new Error('Window selection ended before a window could be read.')) }
          })
          process.stdin.on('error', () => {}); process.stdin.end()
        }) as { cancelled?: boolean; error?: string }
        if (cancelled) return null
        if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('The window picker returned invalid context.')
        if (raw.cancelled === true) return null
        if (raw.error) throw new Error(typeof raw.error === 'string' ? raw.error.slice(0, 300) : 'The window could not be read.')
        return capturedWindowAttachments(raw)
      } finally { child = null; running = false }
    },
  }
}
