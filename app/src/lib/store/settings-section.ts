// One-shot signal: a Settings section to open on. Set by surfaces (e.g. the
// Assistant Sources rail "configure" button) and consumed once by Settings on
// mount, so the user lands directly on the relevant section (Bug 1: configuring a
// source from where it's listed, not the generic Models default).
export const SETTINGS_SECTION_REQUEST_EVENT = 'vontra:settings-section-request'
let pendingSettingsSection: string | null = null
export function requestSettingsSection(section: string): void {
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
