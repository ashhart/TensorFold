import type { ModelQualificationApi } from '../../../shared/model-fit'
/** No model measurements are invented in browser preview. */
export function createMockModelQualificationApi(): ModelQualificationApi {
  return { modelQualificationReports: async () => [], modelQualificationCheck: async () => { throw new Error('Model capability checks require an explicitly selected, loaded desktop model.') }, modelQualificationCancel: async () => {} }
}
