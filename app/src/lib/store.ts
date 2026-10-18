// Thin re-export entry point. The store implementation now lives in ./store/ (domain slices +
// shared helpers under the zustand slice pattern). This file exists so the ~26 importers of
// `'../lib/store'` / `'./store'` (including src/lib/store.test.ts) keep working unchanged.
export { useStore } from './store/index'
export type { AppState, View, ProjectTab } from './store/types'
export { requestSettingsSection, consumeSettingsSection, SETTINGS_SECTION_REQUEST_EVENT, requestConnectorSetup, consumeConnectorSetup } from './store/settings-section'
