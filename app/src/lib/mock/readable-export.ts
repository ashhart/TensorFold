import type { ReadableExportApi } from '../../../shared/readable-export'

/** Native redaction and filesystem export are deliberately unavailable in a browser demo. */
export function createMockReadableExportApi(): ReadableExportApi {
  return {
    async readableExportPreview() { throw new Error('Readable export requires the desktop app, where redaction runs before a native save preview.') },
    async readableExportSave() { return { saved: false, error: 'Native file saving is unavailable in the browser preview.' } },
  }
}
