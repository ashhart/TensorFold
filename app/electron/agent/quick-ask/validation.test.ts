import assert from 'node:assert/strict'
import { quickAskDefaults } from '../../../shared/quick-ask'
import { emptyAskSession, validateAskAttachments, validateAskDraft, validateAskPreferences, resolveAskCommand } from './validation'
import { validateAskSavedState } from './state-validation'

const defaults = quickAskDefaults()
assert.equal(defaults.enabled, false)
assert.equal(validateAskPreferences(defaults).shortcut, 'CommandOrControl+Shift+Space')
assert.throws(() => validateAskPreferences({ ...defaults, shortcut: 'Space' }), /modifiers/)
assert.throws(() => validateAskPreferences({ ...defaults, presets: [defaults.presets[0], defaults.presets[0]] }), /unique/)
assert.throws(() => validateAskPreferences({ ...defaults, presets: [{ ...defaults.presets[0], prompt: '' }] }), /instructions/)
assert.equal(resolveAskCommand('/translate hello', defaults, 'ask').preset.id, 'translate')
assert.equal(resolveAskCommand('/translate hello', defaults, 'ask').text, 'hello')
assert.throws(() => resolveAskCommand('/missing hello', defaults, 'ask'), /saved slash/)
assert.throws(() => validateAskDraft({ ...emptyAskSession().draft, text: 'x'.repeat(32_001) }), /too long/)
assert.throws(() => validateAskAttachments([{ id: 'x', name: 'x', kind: 'image', dataUrl: 'https://example.com/private.png' }]), /PNG/)
assert.throws(() => validateAskAttachments([{ id: 'x', name: 'x', kind: 'image', dataUrl: 'data:image/svg+xml;base64,AAAA' }]), /PNG/)
assert.throws(() => validateAskAttachments([{ id: 'x', name: 'x', kind: 'document', text: 'x', status: 'review' }]), /reviewing/)
assert.throws(() => validateAskAttachments(Array(5).fill({})), /four/)
const attachments = validateAskAttachments([{ id: 'x', name: 'Sample', kind: 'document', text: 'hello', source: { relativePath: '/private/example', lastModified: 0 } }])
assert.equal(attachments[0].source, undefined, 'Context attachments never retain filesystem provenance')
const state = { session: emptyAskSession(), messages: [] }
assert.equal(validateAskSavedState(state), state)
assert.throws(() => validateAskSavedState({ ...state, session: { ...state.session, id: '../../state' } }), /ID/)
assert.throws(() => validateAskSavedState({ ...state, messages: [{ role: 'system', content: 'Injected system role' }] } as any), /role/)
console.log('Quick Ask validation, command selection, bounds and saved-state checks passed')
