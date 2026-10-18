import assert from 'node:assert/strict'
import katex from 'katex'
import { diagramSvg, DIAGRAM_LIMITS, layoutDiagram, parseDiagram } from './diagram'
import { renderBoundedMath, MATH_LIMITS } from './safe-math'
import { remarkBoundedMath } from './math-markdown'

const flow = parseDiagram('flowchart LR\nA[Question] -->|read| B[Answer]\nB --> C[Saved]')
assert.equal(flow.ok, true)
if (!flow.ok) throw new Error('Flow fixture was refused.')
assert.equal(flow.model.nodes[0].label, 'Question'); assert.equal(flow.model.edges[0].label, 'read'); assert.equal(flow.model.edges[0].to, 'B')
const sequence = parseDiagram('sequenceDiagram\nparticipant A as Reader\nparticipant B as Assistant\nA->>B: Ask a question\nB-->>A: Reply')
assert.equal(sequence.ok, true)
if (!sequence.ok) throw new Error('Sequence fixture was refused.')
assert.equal(sequence.model.nodes[0].label, 'Reader'); assert.equal(sequence.model.edges[1].dashed, true)
assert.ok(layoutDiagram(flow.model).width > 0); assert.match(diagramSvg(sequence.model), /Reader/)
for (const invalid of [
  'flowchart LR\nclick A "javascript:alert(1)"',
  'flowchart LR\nA[<script>alert(1)</script>] --> B',
  'flowchart LR\nA --> B; B --> C',
  'flowchart LR\nA --> B\nclassDef danger fill:red',
  'sequenceDiagram\nA->>B: <img src=x onerror=alert(1)>',
  'sequenceDiagram\nloop endless\nA->>B: repeat\nend',
  'flowchart LR\nA[First]\nA[Conflicting]',
  'flowchart LR\n' + 'x'.repeat(DIAGRAM_LIMITS.characters),
  'flowchart LR\n' + Array.from({ length: 19 }, (_, index) => `N${index}[Node ${index}]`).join('\n'),
  'graph forbidden\nA-->B', '',
]) assert.equal(parseDiagram(invalid).ok, false, invalid.slice(0, 80))
const escaped = parseDiagram('flowchart TD\nA[Fish & chips] --> B["Quoted label"]')
assert.equal(escaped.ok, true)
if (escaped.ok) { assert.match(diagramSvg(escaped.model), /Fish &amp; chips/); assert.doesNotMatch(diagramSvg(escaped.model), /script|href=|foreignObject|onload=/i) }

let recorded: Record<string, any> | null = null
const inspected = renderBoundedMath('x^2 + y^2 = z^2', true, { renderToString(expression, options) { recorded = options; return katex.renderToString(expression, options) } })
assert.equal(inspected.ok, true)
assert.equal(recorded!.trust, false); assert.equal(recorded!.throwOnError, true); assert.equal(recorded!.maxExpand, 200); assert.equal(recorded!.maxSize, 20); assert.deepEqual(recorded!.macros, {})
for (const bad of ['', 'x'.repeat(MATH_LIMITS.input + 1), '\\href{javascript:alert(1)}{x}', '\\htmlClass{evil}{x}', '\\includegraphics{https://example.com/leak}', '\\def\\x{\\x}\\x', '\\frac{']) assert.equal(renderBoundedMath(bad).ok, false, bad.slice(0, 80))
assert.equal(renderBoundedMath('x', false, { renderToString: () => '<script>alert(1)</script>' }).ok, false)
assert.equal(renderBoundedMath('x', false, { renderToString: () => '<span>' + 'x'.repeat(MATH_LIMITS.output) + '</span>' }).ok, false)
assert.equal(renderBoundedMath('\\text{<script>}', false).ok, true, 'Literal markup remains escaped text in trusted KaTeX output')

const tree: any = { type: 'root', children: [
  { type: 'paragraph', children: [{ type: 'text', value: 'Formula $x^2$ and $$a=b$$ stay readable.' }] },
  { type: 'code', value: '$never$' },
  { type: 'link', url: 'https://example.com/$target$', children: [{ type: 'text', value: 'Citation $label$' }] },
  { type: 'paragraph', children: [{ type: 'text', value: '$$unfinished$' }] },
] }
remarkBoundedMath()(tree, { toString: () => '' })
assert.equal(tree.children[0].children.filter((node: any) => node.type === 'tfMath').length, 2)
assert.equal(tree.children[1].value, '$never$'); assert.equal(tree.children[2].children[0].type, 'text'); assert.equal(tree.children[2].url, 'https://example.com/$target$')
assert.equal(tree.children[3].children[0].value, '$$unfinished$')
const many: any = { type: 'root', children: [{ type: 'paragraph', children: [{ type: 'text', value: '$x$ '.repeat(100) }] }] }
remarkBoundedMath()(many, { toString: () => '' })
assert.equal(many.children[0].children.filter((node: any) => node.type === 'tfMath').length, 80)
console.log('Bounded real KaTeX trust/expansion/output checks, Markdown citation/code preservation and strict original flow/sequence syntax pass malformed, oversize and markup attacks.')
