import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url), ts = require('typescript'), root = process.cwd()
const module = { exports: {} }, shared = 'src/components/settings/shared.tsx'
const output = ts.transpileModule(readFileSync(shared, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const imported = {
  '../../lib/preferences/useAuthoredText': { useAuthoredText() { throw new Error('Localization hook executed during module initialization') } },
  '../../lib/preferences/language-packs': { authoredText: (_locale, text) => text },
  '../../lib/preferences/display': { currentAuthoredText: text => text },
  '../../lib/store': { useStore: { getState: () => ({ settings: { uiPreferences: { locale: 'en' } } }) } },
  react: { useState() { throw new Error('React state executed during module initialization') } },
  'react/jsx-runtime': { jsx() {}, jsxs() {} }, clsx: { default() {} },
  '../../lib/icons': { Icon() {} }, '../ui': { Button() {}, Field() {}, Input() {}, Toggle() {} },
  '../../lib/cli-models': Object.fromEntries(['CODEX', 'CLAUDE', 'OPENCODE', 'KILO', 'PI', 'CURSOR', 'GEMINI'].map(key => [`${key}_CLI_MODELS`, []])),
  '../../lib/provider-accounts': { modelDisplayName: value => value, providerModelGroups: () => [] },
  '../../lib/agent-builder': { agentBuilderModelSourceCount: () => 0 },
}
assert.doesNotThrow(() => new Function('require', 'module', 'exports', output)(id => { assert.ok(Object.hasOwn(imported, id), `Reviewed initialization import ${id}`); return imported[id] }, module, module.exports), 'Importing actual settings must not call a React hook through static filter/flatMap callbacks')
assert.ok(module.exports.HTTP_CLOUD_PROVIDER_PRESETS.length > 0)
assert.ok(module.exports.HTTP_CLOUD_PROVIDER_PRESETS.every(preset => preset.auth !== 'cli-oauth'))
assert.ok(module.exports.ALL_SETTINGS_SECTIONS.length > 10)
function ownerName(node) {
  if (node.name && ts.isIdentifier(node.name)) return node.name.text
  let parent = node.parent
  while (ts.isCallExpression(parent)) {
    if (!/^(?:React\.)?(?:memo|forwardRef)$/.test(parent.expression.getText())) return ''
    parent = parent.parent
  }
  if (ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) return parent.name.text
  if (ts.isFunctionDeclaration(node) && node.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword)) return 'DefaultComponent'
  return ''
}
const violations = []
function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name)
    if (entry.isDirectory()) { walk(file); continue }
    if (!/\.tsx?$/.test(entry.name) || /\.test\.|test-fixture/.test(entry.name)) continue
    const sf = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, entry.name.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS)
    function visit(node) {
      if (ts.isCallExpression(node) && node.expression.getText(sf) === 'useAuthoredText') {
        let owner = node.parent
        while (owner && !ts.isFunctionLike(owner)) owner = owner.parent
        const name = owner ? ownerName(owner) : ''
        const asynchronous = owner?.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword)
        if (asynchronous || !/^(?:[A-Z]|use[A-Z])/.test(name)) violations.push(`${file.slice(root.length + 1)}:${sf.getLineAndCharacterOfPosition(node.pos).line + 1} ${name || 'callback/module'}${asynchronous ? ' (async)' : ''}`)
      }
      ts.forEachChild(node, visit)
    }; visit(sf)
  }
}
walk(path.join(root, 'src'))
assert.deepEqual(violations, [], 'Localization hooks belong directly to components/custom hooks, never helpers, callbacks or module initialization')
console.log('Actual settings module imports safely and localization hooks have valid component ownership without conditional helper hooks')
