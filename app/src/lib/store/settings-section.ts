import { isConnectorSetupId, type ConnectorSetupId } from '../../../shared/connector-catalog'

// One-shot signal: a Settings section to open on. Set by surfaces (e.g. the
// Assistant Sources rail "configure" button) and consumed once by Settings on
// mount, so the user lands directly on the relevant section (Bug 1: configuring a
// source from where it's listed, not the generic Models default).
export const SETTINGS_SECTION_REQUEST_EVENT = 'vontra:settings-section-request'
let pendingSettingsSection: string | null = null
let pendingConnectorSetup: ConnectorSetupId | null = null
export function requestConnectorSetup(id: ConnectorSetupId): void {
  if (!isConnectorSetupId(id)) throw new Error('Choose a supported connection.')
  pendingConnectorSetup = id
  requestSettingsSection('connectors')
}
export function consumeConnectorSetup(): ConnectorSetupId | null {
  const id = pendingConnectorSetup
  pendingConnectorSetup = null
  return id
}
export function requestSettingsSection(section: string): void {
  if (section !== 'connectors') pendingConnectorSetup = null
  pendingSettingsSection = section
  if (typeof window !== 'undefined') {
    const emit = () => window.dispatchEvent(new CustomEvent(SETTINGS_SECTION_REQUEST_EVENT))
    const emitIfPending = () => { if (pendingSettingsSection === section) emit() }
    emit()
    window.setTimeout(emitIfPending, 0)
    window.setTimeout(emitIfPending, 60)
  }
}
export function consumeSettingsSection(): string | null { const s = pendingSettingsSection; pendingSettingsSection = null; return s }
