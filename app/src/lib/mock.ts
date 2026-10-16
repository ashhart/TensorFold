// Browser demo mode. Active ONLY when the Electron preload bridge is absent
// (i.e. the renderer is opened in a plain browser at the Vite dev URL).
// Lets the full UI be previewed and exercised without the desktop shell.
export { createMockApi } from './mock/api'
