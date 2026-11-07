import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import ts from 'typescript'

const app = new URL('../../../app/', import.meta.url)
const ui = new URL('../', import.meta.url)
const read = (path) => readFile(new URL(path, app), 'utf8')
const theme = await read('src/lib/theme.ts')
const presets = await read('shared/theme-presets.ts')
const styles = await read('src/styles/index.css')
const load = async (source) => {
  const code = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
  }).outputText
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
}
const { buildThemeVars } = await load(theme)
const { PRESETS } = await load(presets)
const clean = styles.replace(/\/\*[\s\S]*?\*\//g, '')
const root = clean.match(/:root\s*\{([^}]+)\}/)?.[1]
const dark = clean.match(/\.dark\s*\{([^}]+)\}/)?.[1]
if (!root || !dark) throw new Error('Desktop theme root or dark tokens missing')
const vars = (text) => [...text.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)]
  .map(([, key, value]) => `  ${key === '--ink' ? '--ink-rgb' : key}:${value.trim()};`)
  .join('\n').replaceAll('var(--ink)', 'var(--ink-rgb)')
const mode = (name, isDark, fallback) => {
  const runtime = buildThemeVars(PRESETS.tensorfold[name], isDark)
  return `${isDark ? ':root[data-theme="dark"]' : ':root'}{\n${vars(fallback)}\n${vars(runtime + ';')}\n}\n`
}
const sourceHash = createHash('sha256').update(theme).update(presets).update(styles).digest('hex')
const css = `// Generated from the desktop TensorFold preset; source SHA-256 ${sourceHash}\n`
  + mode('light', false, root) + mode('dark', true, dark)
// CSS has no single-line comment syntax; source provenance is kept in the JSON receipt.
await writeFile(new URL('src/styles/desktop-theme.css', ui), css.slice(css.indexOf('\n') + 1))
await writeFile(new URL('src/styles/desktop-theme.json', ui), JSON.stringify({
  source: ['app/src/lib/theme.ts', 'app/shared/theme-presets.ts', 'app/src/styles/index.css'],
  sourceHash,
  preset: 'tensorfold',
}, null, 2) + '\n')

const zon = await readFile(new URL('../../../build.zig.zon', import.meta.url), 'utf8')
const version = zon.match(/\.version\s*=\s*"([^"]+)"/)?.[1]
if (!version) throw new Error('Native build version missing')
await writeFile(new URL('../../tensorfold_web/data/native-version.json', import.meta.url), JSON.stringify({ version }) + '\n')
