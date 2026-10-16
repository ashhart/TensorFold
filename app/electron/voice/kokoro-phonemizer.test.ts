import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import path from 'node:path'

const requireLocal = createRequire(import.meta.url)
const pkg = requireLocal('phonemizer/package.json')
assert.equal(pkg.version, '1.2.1-tensorfold.1', 'Kokoro must use the reviewed permissive adapter, not the eSpeak-based upstream binary')
const { phonemize } = requireLocal('phonemizer')
assert.deepEqual(await phonemize("I'm just testing."), ['ˈIm ʤˈʌst tˈɛstɪŋ.'])
const number = (await phonemize('There are twelve files and 123 errors.'))[0]
assert.ok(number.length > 15)
assert.doesNotMatch(number, /\d/)
assert.ok((await phonemize('TensorFold antidisestablishmentarianism local speech.'))[0].length > 10, 'unfamiliar words must use local letter-to-sound rules')
await assert.rejects(() => phonemize('Hello', 'en'), /American English/)
await assert.rejects(() => phonemize('a'.repeat(5001)), /Invalid/)
const adapter = await readFile(requireLocal.resolve('phonemizer'), 'utf8')
assert.doesNotMatch(adapter, /fetch\(|https:\/\/|WebAssembly|espeakng\.worker/)
const headRoot = path.dirname(requireLocal.resolve('@met4citizen/headtts/package.json'))
assert.match(await readFile(path.join(headRoot, 'LICENSE'), 'utf8'), /MIT License/)
assert.match(await readFile(path.join(headRoot, 'dictionaries', 'en-us.txt'), 'utf8'), /Carnegie Mellon University/)
console.log('Pinned GPL-free Kokoro phonemization, dictionary, numbers, punctuation and unknown-word checks passed')
