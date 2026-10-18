// Static imports work offline in desktop, browser and the isolated test runner.
import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { resolve, join } from 'node:path'
const root = resolve(import.meta.dirname, '..', 'src/lib/preferences/packs')
const allowed = new Set(['es', 'es-419', 'zh-Hans', 'zh-Hant', 'ja', 'ko', 'ru', 'pl', 'la', 'hi', 'th', 'vi', 'id'])
const imports = [], entries = [], counts = {}, nativePacks = {}, coverage = {}
const sourceDir = resolve(root, '../sources'), required = new Set(readdirSync(sourceDir).filter(file => file.endsWith('.json')).flatMap(file => JSON.parse(readFileSync(join(sourceDir, file), 'utf8'))))
const nativePhrases = JSON.parse(readFileSync(resolve(root, '../../../../shared/native-ui-phrases.json'), 'utf8'))
let ordinal = 0
for (const locale of readdirSync(root).filter(value => allowed.has(value)).sort()) {
  const files = readdirSync(join(root, locale)).filter(value => /^[a-zA-Z0-9_-]+\.json$/.test(value)).sort(), ids = [], keys = new Set()
  for (const file of files) {
    const data = JSON.parse(readFileSync(join(root, locale, file), 'utf8'))
    if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error(`Invalid language data: ${locale}/${file}`)
    for (const [key, value] of Object.entries(data)) {
      if (!key || typeof value !== 'string' || !value.trim() || key.length > 10_000 || value.length > 20_000) throw new Error(`Invalid phrase: ${locale}/${file}`)
      if (keys.has(key)) throw new Error(`Duplicate phrase: ${locale}/${file}`)
      const placeholders = text => [...text.matchAll(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g)].map(match => match[1]).sort().join(',')
      if (placeholders(key) !== placeholders(value)) throw new Error(`Changed placeholders: ${locale}/${file}: ${key}`)
      keys.add(key)
      if (nativePhrases.includes(key)) (nativePacks[locale] ??= {})[key] = value
    }
    const id = `part${ordinal++}`; imports.push(`import ${id} from './${locale}/${file}'`); ids.push(id)
  }
  if (ids.length) { entries.push(`  '${locale}': Object.freeze(Object.assign(Object.create(null), ${ids.join(', ')})),`); counts[locale] = keys.size; coverage[locale] = { present: [...keys].filter(key => required.has(key)).length, total: required.size, complete: [...required].every(key => keys.has(key)) } }
}
const code = `// Generated from authored bundled packs; run npm run languages:index after editing JSON.\n${imports.join('\n')}\nexport const languagePacks: Readonly<Record<string, Readonly<Record<string, string>>>> = Object.freeze({\n${entries.join('\n')}\n})\nexport const languagePackCoverage: Readonly<Record<string, { present: number; total: number; complete: boolean }>> = Object.freeze(${JSON.stringify(coverage, null, 2)})\n`
const target = join(root, 'index.ts')
if (process.argv.includes('--check')) { if (!existsSync(target) || readFileSync(target, 'utf8') !== code) throw new Error('Bundled language index is stale. Run npm run languages:index.') }
else writeFileSync(target, code)
const nativeTarget = resolve(root, '../../../../shared/native-language-packs.json'), nativeCode = JSON.stringify(nativePacks, null, 2) + '\n'
if (process.argv.includes('--check')) { if (readFileSync(nativeTarget, 'utf8') !== nativeCode) throw new Error('Native language index is stale.') } else writeFileSync(nativeTarget, nativeCode)
console.log(`Bundled language index: ${JSON.stringify(counts)}`)
