import type { PullTimeout } from '../pull-inactivity'

export class PullClock {
  now = 0
  private nextId = 0
  private timers = new Map<number, { at: number; callback: () => void }>()

  schedule: PullTimeout = (callback, ms) => {
    const id = ++this.nextId
    this.timers.set(id, { at: this.now + ms, callback })
    return () => { this.timers.delete(id) }
  }

  advance(ms: number): void {
    this.now += ms
    for (const [id, timer] of this.timers) {
      if (timer.at > this.now) continue
      this.timers.delete(id)
      timer.callback()
    }
  }

  get pending(): number { return this.timers.size }
}
