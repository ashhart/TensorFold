import { spawn } from 'node:child_process'
import path from 'node:path'

export function windowsTaskkill(pid: number, systemRoot = process.env.SystemRoot || 'C:\\Windows'): { command: string; args: string[] } {
  if (!Number.isSafeInteger(pid) || pid <= 0 || pid > 0xffffffff) throw new Error('Invalid owned process ID.')
  if (!path.win32.isAbsolute(systemRoot)) throw new Error('Windows system directory must be absolute.')
  return { command: path.win32.join(systemRoot, 'System32', 'taskkill.exe'), args: ['/PID', String(pid), '/T', '/F'] }
}
export function stopWindowsProcessTree(pid: number, fallback: () => void): void {
  try {
    const { command, args } = windowsTaskkill(pid)
    const killer = spawn(command, args, { windowsHide: true, stdio: 'ignore' })
    const timer = setTimeout(() => { killer.kill(); fallback() }, 5000)
    timer.unref?.()
    killer.once('error', () => { clearTimeout(timer); fallback() })
    killer.once('close', code => { clearTimeout(timer); if (code !== 0) fallback() })
  } catch { fallback() }
}
