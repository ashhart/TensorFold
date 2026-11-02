import { nativeImage } from 'electron'
import mark from '../../../src/assets/tensorfold-mark.png?inline'

export function desktopTrayIcon() {
  const icon = nativeImage.createFromDataURL(mark).resize({ height: 16 })
  if (icon.isEmpty()) throw new Error('TensorFold menu bar icon could not be loaded.')
  icon.setTemplateImage(false)
  return icon
}
