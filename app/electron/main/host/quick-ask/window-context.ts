import { BrowserWindow, desktopCapturer } from 'electron'
import { randomUUID } from 'node:crypto'
import type { QuickAskWindowSelection } from '../../../../shared/quick-ask'
import { computerHelperPath } from '../computer-use-host'
import { createAttachmentTextService } from '../attachment-text'
import { capturedWindowAttachments, inferWindowKind } from '../../../agent/quick-ask/window-context'
import { createNativeAskPicker, supportsNativeAskPicker } from './native-window-picker'

const DENIED = /^(passwords|1password|bitwarden|keychain access|securityagent|keepass|enpass)(?:\b|$)/i

export function createAskContextPicker() {
  const native = createNativeAskPicker(computerHelperPath)
  const text = createAttachmentTextService({ helperPath: computerHelperPath })
  const choices = new Map<string, { sourceId: string; title: string; expiresAt: number }>()
  let listing = false, epoch = 0
  const ownSources = () => new Set(BrowserWindow.getAllWindows().filter(win => !win.isDestroyed()).map(win => win.getMediaSourceId()))
  const windowSources = (width: number, height: number) => desktopCapturer.getSources({ types: ['window'], thumbnailSize: { width, height }, fetchWindowIcons: false })
  return {
    native: supportsNativeAskPicker(),
    get busy() { return listing || native.busy },
    cancel() { if (!native.cancel()) return false; epoch++; choices.clear(); return true },
    async pick(): Promise<QuickAskWindowSelection> {
      if (listing || native.busy) throw new Error('Finish or cancel the current window selection first.')
      const current = ++epoch
      choices.clear()
      if (supportsNativeAskPicker()) {
        const attachments = await native.pick()
        return attachments ? { kind: 'attached', attachments } : { kind: 'cancelled' }
      }
      listing = true
      try {
        const own = ownSources()
        const sources = (await windowSources(240, 150)).filter(source => !own.has(source.id) && !DENIED.test(source.name) && !source.thumbnail.isEmpty()).slice(0, 24)
        if (current !== epoch) return { kind: 'cancelled' }
        if (!sources.length) throw new Error('No readable app windows were offered. Check screen-capture permission or choose a file instead.')
        const windows = sources.map(source => {
          const token = randomUUID(), title = source.name.slice(0, 260)
          choices.set(token, { sourceId: source.id, title, expiresAt: Date.now() + 30_000 })
          return { token, title, thumbnail: source.thumbnail.toDataURL() }
        })
        return { kind: 'choose', windows }
      } finally { listing = false }
    },
    async use(token: string) {
      if (typeof token !== 'string' || token.length > 80) throw new Error('Choose a window from the current picker.')
      const current = epoch
      const selected = choices.get(token)
      choices.clear()
      if (!selected || selected.expiresAt < Date.now()) throw new Error('The window choice expired. Choose it again.')
      const source = (await windowSources(1600, 1000)).find(source => source.id === selected.sourceId)
      if (current !== epoch) throw new Error('Window selection was cancelled.')
      if (!source || source.name.slice(0, 260) !== selected.title || ownSources().has(source.id) || DENIED.test(source.name) || source.thumbnail.isEmpty()) {
        throw new Error('The selected window changed or closed. Choose it again.')
      }
      let image = source.thumbnail, png = image.toPNG()
      for (let attempt = 0; png.length > 1_700_000 && attempt < 6; attempt++) {
        image = image.resize({ width: Math.max(1, Math.floor(image.getSize().width * .75)) }); png = image.toPNG()
      }
      if (png.length > 1_700_000) throw new Error('The selected window image is too large. Choose a smaller window.')
      const result = await text.extract({ name: 'Selected window.png', mime: 'image/png', bytes: png })
      if (current !== epoch) throw new Error('Window selection was cancelled.')
      return capturedWindowAttachments({ app: '', title: selected.title, kind: inferWindowKind(selected.title),
        text: result.ok ? (result.text ?? '').slice(0, 30_000) : '', png: png.toString('base64') })
    },
    dispose() { epoch++; native.cancel(); choices.clear(); text.shutdown() },
  }
}
