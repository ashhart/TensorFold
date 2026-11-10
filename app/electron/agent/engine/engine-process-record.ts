import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'

// The pid of the running engine server, kept on disk so an engine left behind by a crashed app
// can be stopped on the next launch.

function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM' }
}

export class EngineProcessRecord {
  constructor(private readonly appHome: string, private readonly binaryPath: string) {}

  private file(): string {
    return path.join(this.appHome, 'engine', 'serve.json')
  }

  async record(pid: number | null): Promise<void> {
    try {
      if (pid == null) { await fs.rm(this.file(), { force: true }); return }
      await fs.mkdir(path.dirname(this.file()), { recursive: true, mode: 0o700 })
      await fs.writeFile(this.file(), JSON.stringify({ pid, binary: this.binaryPath, owner: process.pid }), { mode: 0o600 })
    } catch { /* bookkeeping only */ }
  }

  // An engine left by a crashed app keeps its model in memory. Stop it only when the app that started
  // it is gone and the recorded pid still runs this exact binary, so a reused pid is never touched.
  async reapStale(): Promise<void> {
    try {
      const record = JSON.parse(await fs.readFile(this.file(), 'utf8'))
      const pid = Number(record?.pid)
      const owner = Number(record?.owner)
      if (!Number.isInteger(pid) || pid <= 1 || record?.binary !== this.binaryPath) return
      if (Number.isInteger(owner) && owner > 1 && owner !== process.pid && processAlive(owner)) return
      const command = await new Promise<string>((resolve) => execFile('/bin/ps', ['-o', 'command=', '-p', String(pid)], (_e, stdout) => resolve(String(stdout ?? '').trim())))
      if (command.startsWith(`${this.binaryPath} serve `)) {
        try { process.kill(-pid, 'SIGTERM') } catch { try { process.kill(pid, 'SIGTERM') } catch { /* gone */ } }
      }
    } catch { /* no record */ }
    await this.record(null)
  }
}
