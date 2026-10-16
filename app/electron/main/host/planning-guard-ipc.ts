import { configurePlanningGuard, previewWithheld } from '../../agent/planning-guard'
import { store } from '../state'
import { handle } from './ipc'

/**
 * Planning-only guard wiring: Settings' confidential terms feed every outbound redaction,
 * and Settings gets a preview of what an external model would receive for a sample text.
 */
export function registerPlanningGuardIpc(): void {
  configurePlanningGuard({ confidentialTerms: () => store.settings?.confidentialTerms })
  handle('guard:previewWithheld', (_e, input: { text?: unknown; terms?: unknown }) => {
    const terms = Array.isArray(input?.terms) ? input.terms.filter((term): term is string => typeof term === 'string').slice(0, 200) : undefined
    return previewWithheld(typeof input?.text === 'string' ? input.text : '', terms)
  })
}
