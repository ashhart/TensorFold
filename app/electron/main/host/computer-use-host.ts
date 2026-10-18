import { app } from 'electron'
import path from 'node:path'
import { ComputerHelper } from '../../agent/computer-use/helper'
import { createComputerUseRunner, type ComputerControl } from '../../agent/computer-use/runner'
import type { ComputerUseInput } from '../../agent/computer-use/actions'
import { APP_BUNDLE_ID } from './app-identity'
import { createComputerIndicator } from './computer-use-indicator'

// Packaged: Contents/Resources/computer-use. Dev: build/computer-use from scripts/build-computer-helper.mjs.
export function computerHelperPath(): string {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'computer-use', 'tensorfold-computer')
    : path.join(app.getAppPath(), 'build', 'computer-use', 'tensorfold-computer')
}

let helper: ComputerHelper | null = null
let runner: ReturnType<typeof createComputerUseRunner> | null = null

export function runComputerUse(input: ComputerUseInput, run: { threadId: string | null; signal?: AbortSignal }, cancel: (threadId: string) => void) {
  helper ??= new ComputerHelper({ binaryPath: computerHelperPath(), ownBundleId: APP_BUNDLE_ID })
  runner ??= createComputerUseRunner({ control: () => helper as unknown as ComputerControl, indicator: createComputerIndicator() })
  const threadId = run.threadId
  return runner.run(input, { threadId, signal: run.signal, cancel: threadId ? () => cancel(threadId) : undefined })
}

export function shutdownComputerUse(): void {
  const current = helper
  helper = null
  void current?.stop().catch(() => {})
}
