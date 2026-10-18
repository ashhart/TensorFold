import type { InputHistoryApi } from '../../shared/input-history'
export function inputHistoryPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): InputHistoryApi {
  return {
    getInputHistory: scope => invoke('input-history:get', scope),
    recallInputHistory: input => invoke('input-history:recall', input),
    clearInputHistory: input => invoke('input-history:clear', input),
  }
}
