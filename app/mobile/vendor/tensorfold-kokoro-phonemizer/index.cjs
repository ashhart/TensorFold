const path = require('node:path')
const { pathToFileURL } = require('node:url')

// Original API adapter for kokoro-js. HeadTTS supplies MIT-licensed English
// phonemization and its BSD-licensed CMU dictionary, without an eSpeak binary.
// Only its local language module is loaded; its server and inference code do not run.
let language
async function englishLanguage() {
  if (!language) language = (async () => {
    const root = path.dirname(require.resolve('@met4citizen/headtts/package.json'))
    const { Language } = await import(pathToFileURL(path.join(root, 'modules', 'language-en-us.mjs')).href)
    const instance = new Language()
    await instance.loadDictionary(path.join(root, 'dictionaries', 'en-us.txt'))
    return instance
  })().catch((error) => { language = undefined; throw error })
  return language
}

async function phonemize(text, locale = 'en-us') {
  if (locale !== 'en-us') throw new Error('This Kokoro installation supports American English pronunciation.')
  if (typeof text !== 'string' || text.length > 5000) throw new Error('Invalid Kokoro speech text.')
  const engine = await englishLanguage()
  return [engine.generate(text).phonemes.join('')]
}

module.exports = { phonemize }
