export type PullTimeout = (callback: () => void, ms: number) => () => void

const scheduleTimeout: PullTimeout = (callback, ms) => {
  const timer = setTimeout(callback, ms)
  timer.unref?.()
  return () => clearTimeout(timer)
}

// Only new cache bytes reset the deadline; console output and existing resumed bytes do not.
export class PullInactivity {
  private bytes = 0
  private cancelTimer: (() => void) | null = null

  constructor(private readonly expired: () => void, private readonly schedule: PullTimeout = scheduleTimeout) {}

  start(bytes: number): void {
    this.stop()
    this.bytes = bytes
    this.cancelTimer = this.schedule(this.expired, 60_000)
  }

  sample(bytes: number): void {
    if (this.cancelTimer && bytes > this.bytes) this.start(bytes)
    else this.bytes = bytes
  }

  stop(): void {
    this.cancelTimer?.()
    this.cancelTimer = null
  }
}
