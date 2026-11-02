import path from 'node:path'

export function localWindowsPath(value: string): string {
  if (!/^[a-z]:[\\/]/i.test(value) || value.includes('\0')) throw new Error('Windows isolation accepts only local absolute filesystem paths.')
  const normalized = path.win32.normalize(value)
  if (normalized.slice(3).includes(':')) throw new Error('Windows isolation refuses alternate data streams and device paths.')
  return normalized
}

export function windowsCommandWords(command: string): string[] {
  return [...new Set(command.match(/[a-zA-Z_][a-zA-Z0-9_.-]*/g) ?? [])].slice(0, 512)
}
