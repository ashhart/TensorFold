import { quickAskDefaults, type QuickAskPreset } from '../../../shared/quick-ask'
export function askPresetLabel(preset: QuickAskPreset, translate: (source: string) => string): string {
  const original = quickAskDefaults().presets.find(item => item.id === preset.id)
  return original?.name === preset.name ? translate(preset.name) : preset.name
}
