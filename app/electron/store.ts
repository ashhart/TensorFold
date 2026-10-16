// Barrel: re-exports the decomposed modules so all importers keep working unchanged.
// The implementation lives in ./store/ (6 focused modules).
export * from './store/paths'
export * from './store/recovery-registry'
export * from './store/app-crypto'
export * from './store/write-infra'
export * from './store/defaults'
export * from './store/persistence'
export * from './store/session-search-index'
