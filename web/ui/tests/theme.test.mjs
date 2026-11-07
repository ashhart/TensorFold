import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import test from 'node:test'
import ts from 'typescript'

const read = (path) => readFile(new URL(path, import.meta.url), 'utf8')
const theme = await read('../../../app/src/lib/theme.ts')
const presets = await read('../../../app/shared/theme-presets.ts')
const appStyles = await read('../../../app/src/styles/index.css')
const load = async (source) => {
  const code = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
  }).outputText
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
}
const { buildThemeVars } = await load(theme)
const { PRESETS } = await load(presets)
const css = await read('../src/styles/desktop-theme.css')
const receipt = JSON.parse(await read('../src/styles/desktop-theme.json'))

test('Web resolves every desktop TensorFold token in both themes', () => {
  for (const [mode, selector] of [['light', ':root'], ['dark', ':root[data-theme="dark"]']]) {
    const start = css.indexOf(selector + '{') + selector.length + 1
    assert.ok(start > selector.length)
    const block = css.slice(start, css.indexOf('}', start))
    const declarations = [...block.matchAll(/(--[\w-]+):([^;]+);/g)]
    const actual = Object.fromEntries(declarations.map(([, key, value]) => [key, value.trim()]))
    for (const declaration of buildThemeVars(PRESETS.tensorfold[mode], mode === 'dark').split(';')) {
      const colon = declaration.indexOf(':')
      const key = declaration.slice(0, colon).replace(/^--ink$/, '--ink-rgb')
      assert.equal(actual[key], declaration.slice(colon + 1), `${mode} ${key}`)
    }
  }
})

test('generated theme records its exact desktop sources', () => {
  const hash = createHash('sha256').update(theme).update(presets).update(appStyles).digest('hex')
  assert.equal(receipt.sourceHash, hash)
  assert.equal(receipt.preset, 'tensorfold')
  assert.ok(!css.includes('Instrument Sans'))
  assert.ok(!css.includes('JetBrains Mono'))
})
