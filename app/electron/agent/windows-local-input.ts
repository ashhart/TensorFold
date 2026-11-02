import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { localWindowsPath } from './windows-sandbox-paths'
import { windowsSandboxHelper } from './windows-sandbox-helper'

export async function windowsLocalInput(input: string): Promise<boolean> {
  const file = localWindowsPath(input)
  try {
    await promisify(execFile)(windowsSandboxHelper().binary, ['--validate-input', file], { timeout: 5000, windowsHide: true })
    return true
  } catch { return false }
}
