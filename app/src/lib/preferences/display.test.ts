import assert from 'node:assert/strict'
import { configureUiLanguageReader, currentUiLanguage, currentIntlLocale, localDateText } from './current-locale'
import { authoredText } from './language-packs'
import { modelCapabilityLabel, modelChoiceDetail, modelFitReason, statusText } from './display'
import { dateBucket, relativeTime, formatDuration } from '../format'
import { groupThreads, threadPlace } from '../thread-row'
let locale = 'en'
const restore = configureUiLanguageReader(() => locale), text = (source: string, params?: any) => authoredText(locale, source, params)
const refusal = 'This model has not passed the required vision, tools check; choose another model or run a new explicit check.'
try {
  assert.equal(modelFitReason(refusal, text), refusal)
  assert.equal(modelCapabilityLabel('Tools: observed · required', text), 'Tools: observed · required')
  locale = 'zh-Hans'
  assert.equal(currentIntlLocale(), 'zh-Hans')
  assert.match(modelFitReason(refusal, text), /\p{Script=Han}/u)
  assert.doesNotMatch(modelFitReason(refusal, text), /\bvision\b|\btools\b/)
  assert.match(modelCapabilityLabel('Tools: observed · required', text), /\p{Script=Han}/u)
  assert.doesNotMatch(modelCapabilityLabel('Tools: observed · required', text), /\bobserved\b|\brequired\b/)
  assert.ok(modelChoiceDetail('On this Mac · Settings', text).endsWith(' · Settings'), 'Account names that match UI words remain original data')
  assert.equal(statusText('PROVIDER_UNKNOWN_STATE', text), 'PROVIDER_UNKNOWN_STATE')
  assert.notEqual(statusText('ready', text), 'ready')
  assert.equal(threadPlace({ kind: 'chat', projectId: 'synthetic', cwd: '' }, { id: 'synthetic', name: 'Settings' }), 'Settings')
  assert.match(dateBucket(Date.now()), /\p{Script=Han}/u)
  assert.match(relativeTime(Date.now() - 120_000), /\p{Script=Han}/u)
  assert.match(formatDuration(120_000), /\p{Script=Han}/u)
  assert.match(groupThreads([{ id: 'synthetic', updatedAt: Date.now() }], new Set())[0].label, /\p{Script=Han}/u)
  locale = 'la'; assert.match(localDateText(Date.now()), /^\d{4}-\d{2}-\d{2}$/)
  locale = 'en'; assert.equal(currentIntlLocale(), undefined); assert.equal(dateBucket(Date.now()), 'Today')
} finally { restore() }
assert.equal(currentUiLanguage(), 'en', 'Locale readers restore without carrying profile data into later renderers')
console.log('Authored model/state/day labels and Intl dates use the current profile language while preserving user/account names, technical states and English compatibility')
