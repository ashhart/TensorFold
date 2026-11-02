import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'

function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM' }
}

export class EngineProcessRecord {
  constructor(private options: { appHome: string; binaryPath: string }) { }
  private pidFile(): string {
    return path.join(this.options.appHome, 'engine', 'serve.json')
  }

  async record(pid: number | null): Promise<void> {
    try {
      if (pid == null) { await fs.rm(this.pidFile(), { force: true }); return }
      await fs.mkdir(path.dirname(this.pidFile()), { recursive: true, mode: 0o700 })
      await fs.writeFile(this.pidFile(), JSON.stringify({ pid, binary: this.options.binaryPath, owner: process.pid }), { mode: 0o600 })
    } catch { }
  }

  async reap(): Promise<void> {
    try {
      const record = JSON.parse(await fs.readFile(this.pidFile(), 'utf8'))
      const pid = Number(record?.pid)
      const owner = Number(record?.owner)
      if (!Number.isInteger(pid) || pid <= 1 || record?.binary !== this.options.binaryPath) return
      if (Number.isInteger(owner) && owner > 1 && owner !== process.pid && processAlive(owner)) return
      const command = await new Promise<string>((resolve) => execFile('/bin/ps', ['-o', 'command=', '-p', String(pid)], (_e, stdout) => resolve(String(stdout ?? '').trim())))
      if (command.startsWith(`${this.options.binaryPath} serve `)) {
        try { process.kill(-pid, 'SIGTERM') } catch { try { process.kill(pid, 'SIGTERM') } catch { } }
      }
    } catch { }
    await this.record(null)
  }
}
