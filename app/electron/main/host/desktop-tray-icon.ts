import { nativeImage } from 'electron'

// Original 32 px transparent fold mark, kept inline so packaged and development builds
// share the same small menu-bar asset without resolving paths outside the signed bundle.
const FOLD_PNG = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABJ0lEQVR4nO1WOw6DMAzlBnAG1DNU6hE69AgMvUGnLqxMPUGnXqAjB+jIzspMZ1am1qhGiqykjp3AxJOeIhF/HsGxSZINemTAG7DHNVsz+RU4AD8GB3y+KM7AjiSm7NAuKk7AxpLsAdzhSvca9AvCAVhbgte4p7VlEfJW3GmxKC3Omu/qqpeSc+wN4xiVTW9MLxEw8w5MhYlT9KOxvAVQIW9g4Zm8QHuaWCzgAhxJoCcwd/jmuG/ajxhHJYALaoITqxYww3asL+AeV+5zBQuY4Cosn4KNImDGEdiSxC0+D4krM1zKdhOwCTBbcIzKpjeGFWAbx5q77eoZ7DieENrd/nVNEaT93XduiOA74aSTUwxuxmv/HUQIGUZRoRlGi6BKfsdfrZ04Gr5gQy1vsUtdCAAAAABJRU5ErkJggg=='

export function desktopTrayIcon() {
  const icon = nativeImage.createFromBuffer(Buffer.from(FOLD_PNG, 'base64'), { scaleFactor: 2 }).resize({ width: 18, height: 18 })
  if (icon.isEmpty()) throw new Error('TensorFold menu bar icon could not be loaded.')
  icon.setTemplateImage(process.platform === 'darwin')
  return icon
}
