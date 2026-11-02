export interface PowerBlockerApi {
  start(type: 'prevent-display-sleep'): number
  stop(id: number): void
  isStarted(id: number): boolean
}

export class DesktopPowerBlocker {
  private id: number | null = null
  constructor(private readonly api: PowerBlockerApi) {}
  start(): void {
    if (this.id !== null && this.api.isStarted(this.id)) return
    this.id = this.api.start('prevent-display-sleep')
    if (!this.api.isStarted(this.id)) { this.id = null; throw new Error('The desktop session refused the sleep inhibitor.') }
  }
  stop(): void {
    const id = this.id
    this.id = null
    if (id !== null) this.api.stop(id)
  }
}
