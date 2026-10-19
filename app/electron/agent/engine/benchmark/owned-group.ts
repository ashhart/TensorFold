import { execFile } from 'node:child_process'
import { readlink, realpath } from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
const env = { PATH: '/usr/bin:/bin', LANG: 'C', LC_ALL: 'C', ...(process.env.TZ ? { TZ: process.env.TZ } : {}) }
export interface OwnedBenchmarkGroup { pid: number; command: string; startedAt: number }
/** A detached benchmark owns a new session/group; inspect survivors before touching that group. */
async function members(group: OwnedBenchmarkGroup): Promise<{ pid: number; command: string; startedAt: number; zombie: boolean }[]> {
  let ids: string
  try { ids = (await exec('/usr/bin/pgrep', ['-g', String(group.pid)], { timeout: 2000, maxBuffer: 64 * 1024, env })).stdout }
  catch (error: any) { if (error.code === 1) return []; throw new Error('Owned process group could not be checked.') }
  const selected = ids.trim().split(/\s+/).filter(Boolean)
  if (selected.length > 32 || selected.some(id => !/^[1-9]\d{0,9}$/.test(id))) throw new Error('Unexpected owned process group inventory.')
  if (!selected.length) return []
  const output = (await exec('/bin/ps', ['-ww', '-o', 'pid=,pgid=,lstart=,stat=,comm=', '-p', selected.join(',')], { timeout: 2000, maxBuffer: 128 * 1024, env })).stdout
  return Promise.all(output.trim().split('\n').filter(Boolean).map(async line => {
    const match = /^\s*(\d+)\s+(\d+)\s+([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s+(\S+)\s+(.+)$/.exec(line)
    if (!match || Number(match[2]) !== group.pid) throw new Error('Owned process group identity changed.')
    const pid = Number(match[1])
    const zombie = match[4].startsWith('Z')
    const command = process.platform === 'linux' && !zombie ? await readlink(`/proc/${pid}/exe`) : match[5].trim()
    return { pid, command, startedAt: Date.parse(match[3]), zombie }
  }))
}
/** The member's real executable: ps comm is a full path on macOS but a 15-char truncated name on Linux. */
async function memberExecutable(member: { pid: number; command: string }): Promise<string | null> {
  if (process.platform === 'linux') return realpath(`/proc/${member.pid}/exe`).catch(() => null)
  return path.resolve(member.command)
}

export async function cleanupOwnedBenchmarkGroup(group: OwnedBenchmarkGroup): Promise<boolean> {
  if (!Number.isSafeInteger(group.pid) || group.pid <= 1) return false
  try {
    const remaining = (await members(group)).filter(member => !member.zombie)
    if (!remaining.length) return true
    const executable = await realpath(group.command)
    for (const member of remaining) {
      if (!Number.isFinite(member.startedAt) || member.startedAt < Math.floor(group.startedAt / 1000) * 1000 - 1000) return false
      if (await memberExecutable(member) !== executable) return false
    }
    try { process.kill(-group.pid, 'SIGKILL') } catch (error: any) { if (error.code !== 'ESRCH') return false }
    const until = Date.now() + 2000
    while (Date.now() < until) {
      if (!(await members(group)).some(member => !member.zombie)) return true
      await new Promise(resolve => setTimeout(resolve, 25))
    }
    return false
  } catch { return false }
}
