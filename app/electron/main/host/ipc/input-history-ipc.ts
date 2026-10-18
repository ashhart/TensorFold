import type { InputHistoryScope } from '../../../../shared/input-history'
import { getInputHistory, recallInputHistory, clearInputHistory } from '../input-history'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export function registerInputHistoryIpc(options: { reserveOperation: () => { release(): void } }): void {
  const run = async <T>(action: () => Promise<T>) => {
    requireSecurityRole('operator', 'use workspace input history')
    const lease = options.reserveOperation()
    try { return await action() } finally { lease.release() }
  }
  handle('input-history:get', (_event, scope: InputHistoryScope) => run(() => getInputHistory(scope)))
  handle('input-history:recall', (_event, input: Parameters<typeof recallInputHistory>[0]) => run(() => recallInputHistory(input)))
  handle('input-history:clear', (_event, input: Parameters<typeof clearInputHistory>[0]) => run(() => clearInputHistory(input)))
}
