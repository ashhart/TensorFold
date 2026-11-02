import { randomUUID } from 'node:crypto'
import type { Attachment } from '../types'
import { quickAskDefaults, type QuickAskDraft, type QuickAskPreferences, type QuickAskSession } from '../../../shared/quick-ask'

export const ASK_TEXT_LIMIT = 32_000
export const ASK_OUTPUT_LIMIT = 64_000
export const ASK_HISTORY_LIMIT = 24
const fail = (message: string): never => { throw new Error(message) }
export function boundedText(value: unknown, max: number, label: string): string {
  if (typeof value !== 'string' || value.length > max) fail(`${label} is invalid or too long.`)
  return value as string
}
export function validateAskAttachments(value: unknown): Attachment[] {
  if (!Array.isArray(value) || value.length > 4) fail('Attach up to four items.')
  let bytes = 0, textChars = 0
  return (value as unknown[]).map(raw => {
    const item = raw as Attachment
    if (!item || !['image', 'document'].includes(item.kind)) fail('Choose an image or text attachment.')
    const id = boundedText(item.id, 120, 'Attachment ID'), name = boundedText(item.name, 200, 'Attachment name')
    if (item.status && item.status !== 'ready') fail('Finish reviewing attachments before asking.')
    if (item.kind === 'document') {
      const text = boundedText(item.text, ASK_TEXT_LIMIT, 'Attached text')
      textChars += text.length
      if (textChars > ASK_TEXT_LIMIT) fail('Attached text is too long. Use a shorter selection.')
      bytes += Buffer.byteLength(text)
      return { id, name, kind: 'document', mime: 'text/plain', text, bytes: Buffer.byteLength(text), status: 'ready' }
    }
    const dataUrl = boundedText(item.dataUrl, 3_000_000, 'Attached image')
    if (!/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(dataUrl)) fail('Choose a valid PNG, JPEG or WebP image.')
    bytes += Buffer.byteLength(dataUrl)
    if (bytes > 2_500_000) fail('Attachments are too large. Use smaller images.')
    return { id, name, kind: 'image', mime: dataUrl.slice(5, dataUrl.indexOf(';')), dataUrl, bytes: Math.floor(dataUrl.length * 0.75), status: 'ready' }
  })
}
export function validateAskDraft(value: QuickAskDraft): QuickAskDraft {
  if (!value || !Number.isSafeInteger(value.revision) || value.revision < 0 || typeof value.web !== 'boolean') fail('The ask draft is invalid.')
  return { revision: value.revision, text: boundedText(value.text, ASK_TEXT_LIMIT, 'Question'), attachments: validateAskAttachments(value.attachments), routeId: value.routeId === null ? null : boundedText(value.routeId, 2000, 'Model route'), presetId: boundedText(value.presetId, 80, 'Preset'), web: value.web }
}
export function validateAskPreferences(value: QuickAskPreferences): QuickAskPreferences {
  if (!value || typeof value.enabled !== 'boolean') fail('Choose a valid shortcut preference.')
  const shortcut = boundedText(value.shortcut, 100, 'Shortcut')
  const parts = shortcut.split('+')
  if (parts.length < 2 || parts.length > 5 || !parts.slice(0, -1).every(part => /^(CommandOrControl|Command|Control|Alt|Shift|Super)$/.test(part)) || !/^(Space|[A-Z0-9]|F(?:[1-9]|1[0-9]|2[0-4]))$/.test(parts.at(-1)!)) fail('Use modifiers and one key, such as CommandOrControl+Shift+Space.')
  if (!Array.isArray(value.presets) || value.presets.length < 1 || value.presets.length > 20) fail('Keep between one and twenty presets.')
  const ids = new Set<string>(), commands = new Set<string>()
  const presets = value.presets.map(preset => {
    const id = boundedText(preset?.id, 80, 'Preset ID'), command = boundedText(preset?.command, 40, 'Preset command')
    if (!/^[a-z0-9-]+$/.test(id) || !/^[\p{L}\p{N}_-]+$/u.test(command) || ids.has(id) || commands.has(command.toLowerCase())) fail('Preset IDs and commands must be unique.')
    ids.add(id); commands.add(command.toLowerCase())
    const name = boundedText(preset.name, 60, 'Preset name'), prompt = boundedText(preset.prompt, 8000, 'Preset instructions')
    if (!name.trim() || !prompt.trim()) fail('Give each preset a name and instructions.')
    return { id, command, name, prompt, routeId: preset.routeId === null ? null : boundedText(preset.routeId, 2000, 'Preset model') }
  })
  return { enabled: value.enabled, shortcut, presets }
}
export function emptyAskSession(): QuickAskSession {
  return { id: randomUUID(), draft: { revision: 0, text: '', attachments: [], routeId: null, presetId: quickAskDefaults().presets[0].id, web: false }, route: null, prompt: null, turns: [] }
}
export function resolveAskCommand(text: string, preferences: QuickAskPreferences, presetId: string) {
  const match = /^\/([^\s]+)(?:\s+([\s\S]*))?$/.exec(text.trimStart())
  const preset = match ? preferences.presets.find(item => item.command.toLowerCase() === match[1].toLowerCase()) : preferences.presets.find(item => item.id === presetId)
  if (!preset) fail(match ? 'Choose a saved slash command or remove the leading slash.' : 'This preset was removed. Choose another preset.')
  return { preset: preset!, text: match ? match[2] ?? '' : text }
}
