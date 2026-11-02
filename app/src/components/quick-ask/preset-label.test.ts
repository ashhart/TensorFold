import assert from 'node:assert/strict'
import { askPresetLabel } from './preset-label'
import { quickAskDefaults } from '../../../shared/quick-ask'
const preset = quickAskDefaults().presets[0], translate = (source: string) => `translated:${source}`
assert.equal(askPresetLabel(preset, translate), 'translated:Ask')
assert.equal(askPresetLabel({ ...preset, name: 'My name' }, translate), 'My name')
assert.equal(askPresetLabel({ ...preset, id: 'custom', name: 'Ask' }, translate), 'Ask')
console.log('Quick Ask translates built-in preset labels and preserves user-authored names')
