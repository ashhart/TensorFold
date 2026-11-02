import assert from 'node:assert/strict'
import { LANGUAGES, languageDirection, normalizeLanguage, responseLanguageInstruction } from '../../../shared/languages'
import { nativeUiText } from '../../../electron/main/host/native-ui-text'
import { authoredText, interfaceLanguages } from './language-packs'
import { languagePackCoverage } from './packs'
import { translate } from './catalogue'
import { flowTranslate } from './flow-catalogue'
import { normalizeUiPreferences } from './model'

assert.equal(LANGUAGES.find(language => language.id === 'pt')?.nativeName, 'Português')
assert.ok(interfaceLanguages().some(language => language.id === 'pt'), 'Portuguese is selectable only with its complete bundled pack')
assert.equal(languagePackCoverage.pt?.complete, true)
for (const alias of ['pt', 'pt-BR', 'pt-PT', 'PT_br', 'pt-AO']) {
  assert.equal(normalizeLanguage(alias), 'pt')
  assert.equal(languageDirection(alias), 'ltr')
  assert.equal(normalizeUiPreferences({ locale: alias }).locale, 'pt')
  assert.match(responseLanguageInstruction(alias), /answers in Portuguese, unless the user explicitly requests another language/)
  assert.equal(translate(alias, 'common.save'), 'Salvar alterações')
  assert.equal(authoredText(alias, 'Language'), 'Idioma')
  assert.equal(nativeUiText(alias, 'Quit TensorFold'), 'Sair do TensorFold')
}
assert.equal(authoredText('pt', 'Gmail reply prepared for review. Nothing was sent.'), 'Resposta do Gmail preparada para revisão. Nada foi enviado.')
assert.equal(authoredText('pt', 'Mailbox reading not granted'), 'Leitura da caixa de e-mail não autorizada')
assert.equal(authoredText('pt', 'Sending not granted'), 'Envio não autorizado')
assert.equal(authoredText('pt', 'Use bundled engine'), 'Usar motor incluído')
assert.equal(nativeUiText('pt', 'Paste as Plain Text'), 'Colar como texto simples')
for (const literal of ['const answer = 42;', 'npm test', 'x-vontra-signature', '-y @modelcontextprotocol/server-filesystem ~/Projects']) {
  assert.equal(authoredText('pt', literal), literal, 'Commands, protocol names and code examples keep their original bytes')
}
assert.match(flowTranslate('pt', 'approval.command'), /Executar este comando/)
const dynamic = authoredText('pt', 'Continue with {mode}', { mode: 'Full <script>fixture</script>\u202e' })
assert.match(dynamic, /Full <script>fixture<\/script>/)
assert.doesNotMatch(dynamic, /\u202e|\{mode\}/)
assert.match(authoredText('pt', '{count} confirmed changes', { count: 1234 }), /1\.234 alterações confirmadas/)
const original = 'HTTP 503: Settings Cancel / original user text'
assert.equal(authoredText('pt', original), original, 'Unknown provider diagnostics and conversation text retain their original bytes')
console.log('Portuguese pack covers regional aliases, preferences, native menus, mail safety wording, engine controls and safe parameters.')
