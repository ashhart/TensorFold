import type { ModelHelperDefaults, ModelHelperRole } from '../../shared/model-fit'

/** Settings contain route identities only; the host resolves credentials and capabilities at use. */
export function normalizeStoredHelperDefaults(input: unknown): ModelHelperDefaults {
  const defaults: ModelHelperDefaults = {}
  if (!input || typeof input !== 'object' || Array.isArray(input)) return defaults
  for (const role of ['naming', 'document_extraction', 'search'] as ModelHelperRole[]) {
    const route = (input as Record<string, any>)[role]
    const valid = (value: unknown) => typeof value === 'string' && value.length > 0 && value.length <= 300 && !/[\u0000-\u001f]/.test(value)
    if (!route || typeof route !== 'object' || Array.isArray(route) || Object.keys(route).some(key => !['providerId', 'accountId', 'model'].includes(key))) continue
    if (valid(route.providerId) && valid(route.model) && (route.accountId === null || valid(route.accountId))) defaults[role] = { providerId: route.providerId, accountId: route.accountId, model: route.model }
  }
  return defaults
}
