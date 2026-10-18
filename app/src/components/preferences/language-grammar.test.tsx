import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import clsx from 'clsx'
import * as languages from '../../../shared/languages'
import { jsx, load, text } from '../reading/reading-test-harness'

// Use the shipped Chinese assets, never a test-only substitute for an unfinished pack.
const chinese: Record<string, string> = {}
for (const file of readdirSync('src/lib/preferences/packs/zh-Hans')) if (file.endsWith('.json')) Object.assign(chinese, JSON.parse(readFileSync(path.join('src/lib/preferences/packs/zh-Hans', file), 'utf8')))
const { authoredText } = load('src/lib/preferences/language-packs.ts', {
  '../../../shared/languages': languages, './packs': { languagePacks: { 'zh-Hans': chinese }, languagePackCoverage: { 'zh-Hans': { complete: true } } },
})
const sourceOnly = 'was s were es branch <script>not translated</script>'
const hook = { useAuthoredText: () => (source: string, params?: Record<string, unknown>) => authoredText('zh-Hans', source, params) }
const { ThreadRow } = load('src/components/sidebar/rows.tsx', {
  'react/jsx-runtime': jsx, clsx, '../../lib/preferences/useAuthoredText': hook, '../../lib/icons': { Icon: 'icon' },
  './constants': { ACTIVE_ROW: '', HOVER_ROW: '' }, './primitives': { InlineEditable: 'editable-title', SidebarSpinner: 'spinner' },
})
for (const count of [4, 1]) {
  const row = ThreadRow({ t: { id: 'synthetic-language-row', title: sourceOnly }, state: 'idle', meta: '', age: '', diff: { files: count, additions: 2, deletions: 1 } })
  const titles: string[] = []
  const inspect = (node: any) => { if (!node || typeof node !== 'object') return; if (node.props?.title) titles.push(node.props.title); const children = node.props?.children; for (const child of Array.isArray(children) ? children.flat(Infinity) : [children]) inspect(child) }
  inspect(row)
  const expected = authoredText('zh-Hans', 'Files changed: {count}', { count })
  assert.match(expected, /\p{Script=Han}/u)
  assert.ok(titles.includes(expected), `Actual sidebar title must render the complete Chinese count without an English plural suffix: ${titles.join('|')}`)
}
const { TurnOutcome } = load('src/components/turn/TurnOutcome.tsx', { 'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': hook, '../../lib/icons': { Icon: 'icon' } })
for (const errors of [1, 2]) {
  const view = TurnOutcome({ turn: { work: [{ runReceipt: { verdict: 'failed', errors, warnings: [] } }] }, onShowActivity() {} })
  const expected = authoredText('zh-Hans', 'Recorded errors: {count}. Verify the result before relying on the answer.', { count: errors })
  assert.match(expected, /\p{Script=Han}/u)
  assert.ok(text(view).includes(expected))
  assert.doesNotMatch(text(view), /\b(?:was|were)\b/)
}
const { AssistantHeader } = load('src/components/assistant/AssistantHeader.tsx', {
  'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': hook,
  '../../lib/brand': { BrandIcon: 'brand', connectorBrand: () => 'synthetic' }, '../../lib/icons': { Icon: 'icon' }, '../ui': { Button: 'button' },
  './mappings': { sourceName: () => sourceOnly, timeAgo: () => 'synthetic-time' },
})
const expectedSignals = authoredText('zh-Hans', 'Open signals: {count} · urgent: {urgent}', { count: 3, urgent: 2 })
assert.match(expectedSignals, /\p{Script=Han}/u)
assert.ok(text(AssistantHeader({ snapshot: null, selectedSourceRow: null, selectedItem: null, openSignals: 3, urgentCount: 2 })).includes(expectedSignals))

// Only output-bearing branches are inspected; enum comparisons and user/provider values remain data.
const violations: string[] = []
function rawFragments(node: ts.Node): string[] {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return /[A-Za-z]/.test(node.text) ? [node.text] : []
  if (ts.isTemplateExpression(node)) return [node.head.text, ...node.templateSpans.flatMap(span => [...rawFragments(span.expression), span.literal.text])].filter(value => /[A-Za-z]/.test(value))
  if (ts.isConditionalExpression(node)) return [...rawFragments(node.whenTrue), ...rawFragments(node.whenFalse)]
  if (ts.isParenthesizedExpression(node) || ts.isAsExpression(node)) return rawFragments(node.expression)
  if (ts.isBinaryExpression(node) && [ts.SyntaxKind.BarBarToken, ts.SyntaxKind.QuestionQuestionToken].includes(node.operatorToken.kind)) return [...rawFragments(node.left), ...rawFragments(node.right)]
  if (ts.isCallExpression(node) && node.expression.getText() === 'String') return node.arguments.flatMap(rawFragments)
  return []
}
function scan(directory: string) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name)
    if (entry.isDirectory()) { scan(file); continue }
    if (!file.endsWith('.tsx') || file.includes('.test.')) continue
    const ast = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
    const visit = (node: ts.Node) => {
      if (ts.isCallExpression(node) && ['tx', 'authoredText'].includes(node.expression.getText(ast))) {
        const params = node.arguments.at(-1)
        if (params && ts.isObjectLiteralExpression(params)) for (const property of params.properties) if (ts.isPropertyAssignment(property)) {
          const fragments = rawFragments(property.initializer)
          if (fragments.length) violations.push(`${file}:${ast.getLineAndCharacterOfPosition(property.getStart(ast)).line + 1} ${fragments.join('|')}`)
        }
      }
      ts.forEachChild(node, visit)
    }
    visit(ast)
  }
}
scan('src/components')
assert.deepEqual(violations, [], 'Authored English suffixes, words and sentence fragments cannot be interpolated as untranslatable data')
assert.equal(authoredText('zh-Hans', '{value0}', { value0: sourceOnly }).replace(/[\u2068\u2069]/g, ''), sourceOnly, 'Arbitrary user text is never translated as a workaround')
console.log('Actual Chinese sidebar, outcome and signal components preserve numeric meaning without English grammar leakage; authored interpolation branches are guarded and raw user text remains original.')
