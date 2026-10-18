import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, mkdir, open, rename, rm } from 'node:fs/promises'
import path from 'node:path'
import { Worker } from 'node:worker_threads'
import type { KokoroStatus, KokoroSynthesisInput } from '../agent/types/kokoro'
import type { VoiceSynthesisResult } from '../agent/types/connectors'
import { KOKORO_ASSETS, KOKORO_DEFAULT_VOICE, KOKORO_REVISION, KOKORO_VOICES, type KokoroAsset } from './kokoro-catalog'

export interface KokoroServiceOptions {
  root: string
  workerFile: string
  fetchImpl?: typeof fetch
  assets?: KokoroAsset[]
  makeWorker?: (file: string, options: { workerData: { modelDirectory: string } }) => Worker
  synthesisTimeoutMs?: number
}

// The installer accepts only our pinned catalog. The renderer never supplies a URL,
// destination, runtime, executable, or model revision.
export class KokoroService {
  private readonly assets: KokoroAsset[]
  private readonly modelDirectory: string
  private current: KokoroStatus
  private installTask: Promise<KokoroStatus> | null = null
  private installAbort: AbortController | null = null
  private worker: Worker | null = null
  private active: { id: string; finish: (result: VoiceSynthesisResult) => void } | null = null
  private idleTimer: ReturnType<typeof setTimeout> | undefined
  private fingerprints = new Map<string, string>()
  private checking: Promise<boolean> | null = null
  private removing = false
  private cancelledRequests = new Set<string>()

  constructor(private readonly options: KokoroServiceOptions) {
    this.assets = options.assets ?? KOKORO_ASSETS
    this.modelDirectory = path.join(options.root, KOKORO_REVISION)
    this.current = {
      state: 'missing', installed: false, downloadedBytes: 0,
      totalBytes: this.assets.reduce((sum, asset) => sum + asset.bytes, 0),
      message: 'Download Kokoro once to read replies offline.', voices: KOKORO_VOICES,
      modelRevision: KOKORO_REVISION,
    }
  }

  private snapshot(): KokoroStatus { return { ...this.current, voices: this.current.voices.map((voice) => ({ ...voice })) } }

  private async ensureDirectory(directory: string): Promise<void> {
    await mkdir(directory, { recursive: true, mode: 0o700 })
    const root = path.resolve(this.options.root)
    const relative = path.relative(root, directory)
    if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Invalid Kokoro model directory.')
    let cursor = root
    for (const piece of ['', ...relative.split(path.sep).filter(Boolean)]) {
      if (piece) cursor = path.join(cursor, piece)
      const info = await lstat(cursor)
      if (info.isSymbolicLink() || !info.isDirectory()) throw new Error('Kokoro model directories must not be symbolic links.')
    }
  }

  private async valid(asset: KokoroAsset): Promise<boolean> {
    const file = path.join(this.modelDirectory, asset.path)
    try {
      const info = await lstat(file)
      if (info.isSymbolicLink() || !info.isFile() || info.size !== asset.bytes) return false
      const fingerprint = `${info.dev}:${info.ino}:${info.size}:${info.mtimeMs}:${info.ctimeMs}`
      if (this.fingerprints.get(file) === fingerprint) return true
      const hash = createHash('sha256')
      for await (const chunk of createReadStream(file)) hash.update(chunk)
      if (hash.digest('hex') !== asset.sha256) return false
      this.fingerprints.set(file, fingerprint)
      return true
    } catch { return false }
  }

  private async checkInstalled(): Promise<boolean> {
    if (this.checking) return this.checking
    this.checking = (async () => {
      try {
        const info = await lstat(this.modelDirectory)
        if (info.isSymbolicLink() || !info.isDirectory()) return false
        for (const asset of this.assets) {
          const parent = await lstat(path.dirname(path.join(this.modelDirectory, asset.path)))
          if (parent.isSymbolicLink() || !parent.isDirectory() || !await this.valid(asset)) return false
        }
        return true
      } catch { return false }
    })()
    try { return await this.checking } finally { this.checking = null }
  }

  async status(): Promise<KokoroStatus> {
    if (this.installTask) return this.snapshot()
    const installed = await this.checkInstalled()
    this.current.installed = installed
    if (installed) {
      this.current.state = 'ready'; this.current.downloadedBytes = this.current.totalBytes
      this.current.message = 'Ready for offline playback.'; delete this.current.error
    } else if (this.current.state === 'ready') {
      this.current.state = 'error'; this.current.message = 'Kokoro model files changed or are missing. Reinstall to repair them.'
      this.current.error = this.current.message; this.stopWorker()
    }
    return this.snapshot()
  }

  install(): Promise<KokoroStatus> {
    if (this.installTask) return this.installTask
    if (this.removing) return Promise.resolve({ ...this.snapshot(), error: 'Wait for Kokoro removal to finish.' })
    this.installAbort = new AbortController()
    this.installTask = this.download(this.installAbort.signal).finally(() => { this.installTask = null; this.installAbort = null })
    return this.installTask
  }

  private async download(signal: AbortSignal): Promise<KokoroStatus> {
    this.current = { ...this.current, state: 'downloading', installed: false, downloadedBytes: 0, message: 'Downloading Kokoro...', error: undefined }
    let completed = 0
    try {
      this.stopWorker()
      await this.ensureDirectory(this.modelDirectory)
      for (const asset of this.assets) {
        signal.throwIfAborted()
        const file = path.join(this.modelDirectory, asset.path)
        await this.ensureDirectory(path.dirname(file))
        if (await this.valid(asset)) { completed += asset.bytes; this.current.downloadedBytes = completed; continue }
        const partial = `${file}.part`
        await rm(partial, { force: true })
        try {
          const response = await (this.options.fetchImpl ?? fetch)(asset.url, {
            signal: AbortSignal.any([signal, AbortSignal.timeout(5 * 60_000)]),
            redirect: 'follow', credentials: 'omit', referrerPolicy: 'no-referrer',
          })
          if (!response.ok || !response.body) { await response.body?.cancel(); throw new Error(`Download returned HTTP ${response.status}.`) }
          const declared = Number(response.headers.get('content-length'))
          if (declared && declared !== asset.bytes) { await response.body.cancel(); throw new Error('The Kokoro download size did not match its manifest.') }
          const handle = await open(partial, 'wx', 0o600)
          const reader = response.body.getReader()
          const hash = createHash('sha256')
          let bytes = 0
          try {
            while (true) {
              signal.throwIfAborted()
              const { done, value } = await reader.read()
              if (done) break
              bytes += value.byteLength
              if (bytes > asset.bytes) throw new Error('The Kokoro download exceeded its expected size.')
              hash.update(value); await handle.writeFile(value)
              this.current.downloadedBytes = completed + bytes
            }
            this.current.state = 'verifying'; this.current.message = 'Verifying Kokoro download...'
            if (bytes !== asset.bytes || hash.digest('hex') !== asset.sha256) throw new Error('Kokoro integrity verification failed. Retry the download.')
            await handle.sync()
          } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); await handle.close() }
          signal.throwIfAborted()
          await rename(partial, file)
        } finally { await rm(partial, { force: true }) }
        completed += asset.bytes; this.current.downloadedBytes = completed
        this.current.state = 'downloading'; this.current.message = 'Downloading Kokoro...'
      }
      signal.throwIfAborted()
      if (!await this.checkInstalled()) throw new Error('Kokoro installation could not be verified. Retry the download.')
      this.current.state = 'ready'; this.current.installed = true; this.current.message = 'Ready for offline playback.'
    } catch (error) {
      const cancelled = signal.aborted
      this.current.state = cancelled ? 'missing' : 'error'
      this.current.message = cancelled ? 'Download cancelled. Retry to continue from verified files.' : 'Kokoro setup failed. Check your connection and retry.'
      const code = (error as NodeJS.ErrnoException)?.code
      this.current.error = cancelled ? undefined : code === 'ENOSPC' ? 'There is not enough disk space for Kokoro.' : error instanceof Error && /integrity|expected size|manifest|symbolic links/.test(error.message) ? error.message : this.current.message
    }
    return this.snapshot()
  }

  async cancelInstall(): Promise<KokoroStatus> { this.installAbort?.abort(); if (this.installTask) await this.installTask; return this.status() }

  async remove(): Promise<KokoroStatus> {
    this.removing = true
    try {
      await this.cancelInstall(); this.stopWorker()
      await rm(this.modelDirectory, { recursive: true, force: true }); this.fingerprints.clear()
      this.current.state = 'missing'; this.current.installed = false; this.current.downloadedBytes = 0
      this.current.message = 'Kokoro removed. Download it again whenever you need it.'; delete this.current.error
      return this.snapshot()
    } finally { this.removing = false }
  }

  private stopWorker(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer)
    const worker = this.worker; this.worker = null
    if (worker) void worker.terminate()
    this.active?.finish({ ok: false, error: 'Speech cancelled.' }); this.active = null
  }

  cancelSpeech(requestId: string): boolean {
    if (/^[a-zA-Z0-9-]{1,80}$/.test(requestId)) {
      this.cancelledRequests.add(requestId)
      if (this.cancelledRequests.size > 200) this.cancelledRequests.delete(this.cancelledRequests.values().next().value!)
    }
    if (this.active?.id !== requestId) return false
    this.stopWorker(); return true
  }

  async synthesize(input: KokoroSynthesisInput): Promise<VoiceSynthesisResult> {
    return this.generate(input)
  }

  async stream(input: KokoroSynthesisInput, onChunk: (wav: Buffer, index: number) => Promise<void>, signal: AbortSignal): Promise<VoiceSynthesisResult> {
    if (signal.aborted) return { ok: false, error: 'Speech cancelled.' }
    if (this.active) return { ok: false, error: 'Kokoro is already reading. Wait for playback to finish.' }
    const abort = () => this.cancelSpeech(input.requestId)
    signal.addEventListener('abort', abort, { once: true })
    try { return await this.generate(input, onChunk, signal) }
    finally { signal.removeEventListener('abort', abort) }
  }

  private async generate(input: KokoroSynthesisInput, onChunk?: (wav: Buffer, index: number) => Promise<void>, signal?: AbortSignal): Promise<VoiceSynthesisResult> {
    if (!input || typeof input.text !== 'string' || !input.text.trim()) return { ok: false, error: 'Nothing to read.' }
    if (input.text.length > 5000) return { ok: false, error: 'Read up to 5,000 characters at a time, or turn on Brief before reading.' }
    if (typeof input.requestId !== 'string' || !/^[a-zA-Z0-9-]{1,80}$/.test(input.requestId)) return { ok: false, error: 'Invalid speech request.' }
    const voice = input.voice || KOKORO_DEFAULT_VOICE
    if (!KOKORO_VOICES.some((item) => item.id === voice)) return { ok: false, error: 'Choose a supported Kokoro voice in Voice settings.' }
    if (this.removing || this.installTask || !await this.checkInstalled()) return { ok: false, error: 'Download or repair Kokoro in Settings > Voice before reading.' }
    if (this.cancelledRequests.delete(input.requestId)) return { ok: false, error: 'Speech cancelled.' }
    if (signal?.aborted) return { ok: false, error: 'Speech cancelled.' }
    // A watch read never interrupts desktop playback, including an overlapping install check.
    if (onChunk && this.active) return { ok: false, error: 'Kokoro is already reading. Wait for playback to finish.' }
    this.stopWorker()
    const rate = Number.isFinite(input.rate) ? Math.max(0.7, Math.min(2.5, input.rate!)) : 1
    return new Promise((resolve) => {
      let timeout: ReturnType<typeof setTimeout>
      let worker: Worker
      try { worker = (this.options.makeWorker ?? ((file, options) => new Worker(file, options)))(this.options.workerFile, { workerData: { modelDirectory: this.modelDirectory } }) }
      catch { resolve({ ok: false, error: 'Kokoro could not start. Reopen TensorFold and retry.' }); return }
      this.worker = worker
      const finish = (result: VoiceSynthesisResult) => {
        if (this.active?.finish !== finish) return
        clearTimeout(timeout); this.active = null
        // Unload after each read: cancellation kills native inference immediately and
        // unused voice models never occupy memory alongside the user's chat model.
        if (this.worker === worker) { this.worker = null; void worker.terminate() }
        resolve(result)
      }
      this.active = { id: input.requestId, finish }
      let chunkInFlight = false, nextChunk = 0
      worker.on('message', (result: VoiceSynthesisResult & { requestId?: string; chunk?: boolean; index?: number }) => {
        if (result.requestId !== input.requestId || this.active?.finish !== finish) return
        if (result.chunk) {
          if (!onChunk || chunkInFlight || result.index !== nextChunk || typeof result.audioBase64 !== 'string' || result.audioBase64.length > 4 * 1024 * 1024) {
            finish({ ok: false, error: 'Kokoro returned an invalid audio chunk.' }); return
          }
          const wav = Buffer.from(result.audioBase64, 'base64')
          if (wav.length < 44 || wav.length > 3 * 1024 * 1024 || wav.toString('ascii', 0, 4) !== 'RIFF') { finish({ ok: false, error: 'Kokoro returned an invalid audio chunk.' }); return }
          chunkInFlight = true
          void onChunk(wav, nextChunk).then(() => {
            if (this.active?.finish !== finish || signal?.aborted) return
            chunkInFlight = false; nextChunk++
            worker.postMessage({ requestId: input.requestId, ack: result.index })
          }).catch(() => finish({ ok: false, error: 'Speech playback stopped.' }))
          return
        }
        finish(result)
      })
      worker.once('error', () => finish({ ok: false, error: 'Kokoro could not start. Reinstall its model and retry.' }))
      worker.once('exit', () => finish({ ok: false, error: 'Kokoro stopped before speech was ready. Retry the read.' }))
      timeout = setTimeout(() => { finish({ ok: false, error: 'Kokoro took too long. Try a shorter passage.' }) }, this.options.synthesisTimeoutMs ?? 120_000)
      worker.postMessage({ ...input, text: input.text.trim(), voice, rate, ...(onChunk ? { stream: true } : {}) })
    })
  }

  dispose(): void { this.installAbort?.abort(); this.stopWorker() }
}
