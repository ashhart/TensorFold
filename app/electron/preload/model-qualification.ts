import type { ModelQualificationApi } from '../../shared/model-fit'
export function modelQualificationPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): ModelQualificationApi {
  return { modelQualificationReports: () => invoke('modelQualification:reports'), modelQualificationCheck: input => invoke('modelQualification:check', input), modelQualificationCancel: requestId => invoke('modelQualification:cancel', requestId) }
}
