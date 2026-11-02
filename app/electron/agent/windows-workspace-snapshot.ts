import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { windowsSandboxHelper } from './windows-sandbox-helper'
import { localWindowsPath } from './windows-sandbox-paths'

export async function snapshotWindowsWorkspace(source: string, destination: string, inputLinks: boolean): Promise<void> {
  try {
    await promisify(execFile)(windowsSandboxHelper().binary,
      [inputLinks ? '--export-input' : '--export-staged', localWindowsPath(source), localWindowsPath(destination)],
      { timeout: 120000, windowsHide: true, maxBuffer: 64000 })
  } catch (error: any) {
    throw new Error(`Windows workspace snapshot refused: ${String(error.stderr || error.message).trim()}`)
  }
}
