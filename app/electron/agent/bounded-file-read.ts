import { StringDecoder } from 'node:string_decoder'

export const WORKSPACE_READ_MAX_BYTES = 8 * 1024 * 1024
export const WORKSPACE_READ_TIMEOUT_MS = 15_000
export interface WorkspaceReadOptions { signal?: AbortSignal; timeoutMs?: number; maxBytes?: number }
export interface WorkspaceRangeOptions extends WorkspaceReadOptions { offset: number; limit: number; maxChars: number }
export interface WorkspaceTextRange { text: string; lines: number; bounded: boolean; longLine: boolean; nextOffset: number; totalLines?: number }

export interface FileReadBudget {
  signal: AbortSignal
  maxBytes: number
  wait<T>(promise: Promise<T>): Promise<T>
}

export async function withFileReadBudget<T>(options: WorkspaceReadOptions, action: (budget: FileReadBudget) => Promise<T>): Promise<T> {
  const controller = new AbortController()
  const abort = () => controller.abort(new Error('File read aborted.'))
  const timeout = Math.max(1, Math.min(WORKSPACE_READ_TIMEOUT_MS, options.timeoutMs ?? WORKSPACE_READ_TIMEOUT_MS))
  const timer = setTimeout(() => controller.abort(new Error('File read timed out.')), timeout)
  options.signal?.addEventListener('abort', abort, { once: true })
  if (options.signal?.aborted) abort()
  const budget: FileReadBudget = {
    signal: controller.signal,
    maxBytes: Math.max(1, Math.min(WORKSPACE_READ_MAX_BYTES, Math.floor(options.maxBytes ?? WORKSPACE_READ_MAX_BYTES))),
    wait<T>(promise: Promise<T>): Promise<T> {
      return new Promise<T>((resolve, reject) => {
        const cancel = () => reject(controller.signal.reason)
        if (controller.signal.aborted) cancel()
        else controller.signal.addEventListener('abort', cancel, { once: true })
        promise.then(resolve, reject).finally(() => controller.signal.removeEventListener('abort', cancel))
      })
    },
  }
  try {
    if (controller.signal.aborted) throw controller.signal.reason
    return await budget.wait(action(budget))
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', abort)
  }
}

export interface OpenedTextFile {
  stat(): Promise<{ isFile: boolean; size: number }>
  read(buffer: Buffer, length: number, position: number): Promise<number>
  close(): Promise<void>
}

/** Keeps only the selected lines and stops once the caller's output budget is filled. */
class LineSelection {
  private decoder = new StringDecoder('utf8')
  private line = 1
  private current = ''
  private selected: string[] = []
  private chars = 0
  private ended = false
  private bounded = false
  private longLine = false
  constructor(private options: WorkspaceRangeOptions) {}
  get done() { return this.ended }

  feed(buffer: Buffer): void { this.consume(this.decoder.write(buffer)) }
  private consume(text: string): void {
    let position = 0
    while (position < text.length && !this.ended) {
      const newline = text.indexOf('\n', position)
      const end = newline === -1 ? text.length : newline
      if (this.line >= this.options.offset) {
        const prefix = `${String(this.line).padStart(5)}\t`
        const remaining = this.options.maxChars - this.chars - prefix.length - (this.selected.length ? 1 : 0)
        const piece = text.slice(position, end)
        if (this.current.length + piece.length > remaining) {
          this.bounded = true
          this.ended = true
          if (!this.selected.length) {
            this.longLine = true
            this.selected.push(prefix + (this.current + piece).slice(0, Math.max(0, remaining)))
          }
          return
        }
        this.current += piece
      }
      if (newline === -1) return
      this.finishLine()
      this.line++
      position = newline + 1
    }
  }
  private finishLine(): void {
    if (this.line >= this.options.offset) {
      const numbered = `${String(this.line).padStart(5)}\t${this.current}`
      this.chars += numbered.length + (this.selected.length ? 1 : 0)
      this.selected.push(numbered)
      this.current = ''
      if (this.selected.length >= this.options.limit) this.ended = true
    }
  }
  result(eof: boolean): WorkspaceTextRange {
    if (eof && !this.ended) {
      this.consume(this.decoder.end())
      if (!this.ended) this.finishLine()
    }
    return { text: this.selected.join('\n'), lines: this.selected.length, bounded: this.bounded, longLine: this.longLine, nextOffset: this.options.offset + this.selected.length, ...(eof ? { totalLines: this.line } : {}) }
  }
}

export async function readOpenedTextFile(file: OpenedTextFile, budget: FileReadBudget, range?: WorkspaceRangeOptions): Promise<Buffer | WorkspaceTextRange> {
  try {
    const stat = await budget.wait(file.stat())
    if (!stat.isFile || !Number.isSafeInteger(stat.size) || stat.size < 0) throw new Error('Refusing to read a non-regular workspace file.')
    if (!range && stat.size > budget.maxBytes) throw new Error(`Workspace file exceeds the ${budget.maxBytes}-byte read limit.`)
    const selection = range ? new LineSelection(range) : null
    const chunks: Buffer[] = []
    let position = 0
    while (!selection?.done) {
      if (position >= budget.maxBytes) {
        // Reaching the opening stat size is not EOF: a regular file may grow.
        const extra = await budget.wait(file.read(Buffer.alloc(1), 1, position))
        if (!extra) return selection ? selection.result(true) : Buffer.concat(chunks, position)
        throw new Error(`Workspace read exceeded the ${budget.maxBytes}-byte scan limit; use a narrower file or search.`)
      }
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, budget.maxBytes - position))
      const bytes = await budget.wait(file.read(buffer, buffer.length, position))
      if (!Number.isInteger(bytes) || bytes < 0 || bytes > buffer.length) throw new Error('Invalid workspace read length.')
      if (!bytes) return selection ? selection.result(true) : Buffer.concat(chunks, position)
      position += bytes
      if (selection) selection.feed(buffer.subarray(0, bytes))
      else chunks.push(buffer.subarray(0, bytes))
    }
    return selection!.result(false)
  } finally {
    // Cancellation must not wait for an unresponsive remote close acknowledgement.
    const closing = file.close().catch(() => {})
    if (!budget.signal.aborted) await budget.wait(closing)
  }
}
