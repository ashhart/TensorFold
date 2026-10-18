import type { ReadableExportApi } from '../../shared/readable-export'
export function readableExportPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): ReadableExportApi {
  return { readableExportPreview: input => invoke('readableExport:preview', input), readableExportSave: input => invoke('readableExport:save', input) }
}
