import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import type { Link, Parent, Root, RootContent } from 'mdast'
import { remarkCitationAutolinks } from './markdown-citations'

const target = 'https://ziglang.org/download/0.15.1/release-notes.html'
const observedAnswer = `Source 【${target}】`
const render = (source: string, normalize = true) => renderToStaticMarkup(React.createElement(ReactMarkdown, { remarkPlugins: normalize ? [remarkGfm, remarkCitationAutolinks] : [remarkGfm] }, source))
const parse = (source: string): Root => {
  const parser = unified().use(remarkParse).use(remarkGfm).use(remarkCitationAutolinks)
  return parser.runSync(parser.parse(source), source) as Root
}
const links = (tree: Root): Link[] => {
  const found: Link[] = []
  const visit = (node: Root | RootContent) => {
    if (node.type === 'link') found.push(node)
    if ('children' in node) (node as Parent).children.forEach(visit)
  }
  visit(tree)
  return found
}

assert.ok(render(observedAnswer, false).includes(`${target}%E3%80%91`), 'the actual GFM renderer reproduces the broken native citation target')
assert.equal(links(parse(observedAnswer))[0]?.url, target, 'only the URL belongs in the auto-linked citation destination')
assert.equal(render(observedAnswer), `<p>Source 【<a href="${target}">${target}</a>】</p>`, 'the visible citation brackets must stay outside the clickable URL')

for (const [open, close] of [['【', '】'], ['〖', '〗'], ['〔', '〕'], ['［', '］'], ['「', '」'], ['『', '』'], ['〈', '〉'], ['《', '》'], ['（', '）']]) {
  const source = `${open}${target}${close}.`
  assert.equal(links(parse(source))[0]?.url, target, `${open}${close} citation punctuation is excluded from an automatic target`)
  assert.ok(render(source).endsWith(`</a>${close}.</p>`), 'punctuation is preserved as visible text')
  const balanced = `${target}/${open}valid-path${close}`
  assert.equal(links(parse(balanced))[0]?.url, balanced, 'balanced punctuation inside a URL is preserved')
}

const nested = `【${target}/【section】】`
assert.equal(links(parse(nested))[0]?.url, `${target}/【section】`, 'only the unmatched outer closer is removed')
assert.ok(render(nested).includes('%E3%80%90section%E3%80%91'), 'balanced URL punctuation retains its encoded destination')
assert.equal(render(`〔【${target}】〕`), `<p>〔【<a href="${target}">${target}</a>】〕</p>`, 'multiple unmatched citation closers retain their visible order')

for (const suffix of ['%E3%80%91', '%e3%80%91', '?q=%E3%80%91', '/%E3%80%90section】', '/【section%E3%80%91', '/Thing_(details)', '/array[0]', '/fragment#section']) {
  const source = `${target}${suffix}`
  assert.equal(render(source), render(source, false), 'valid percent encodings and balanced ASCII punctuation are unchanged')
}
assert.equal(links(parse(`【${target}/%E3%80%90section】】`))[0]?.url, `${target}/%E3%80%90section】`, 'encoded openers are balanced without changing their bytes')
assert.equal(render('【www.example.test/docs】'), render('【www.example.test/docs】', false), 'positionless GFM www links are preserved when source provenance cannot be established')
assert.equal(links(parse(`【${target}?q=a&amp;b=c】`))[0]?.url, `${target}?q=a&amp;b=c`, 'entity spelling within GFM literal URLs is preserved')

for (const source of [
  `[explicit](${target}】)`,
  `[${target}】](${target}】)`,
  `<${target}】>`,
  `[reference][source]\n\n[source]: ${target}】`,
  `\`${observedAnswer}\``,
  `\`\`\`text\n${observedAnswer}\n\`\`\``,
  `    ${observedAnswer}`,
]) {
  assert.equal(render(source), render(source, false), 'explicit links, references and code must not be normalized')
}

const mixed = `**Evidence 【${target}】**\n\n| Source |\n| --- |\n| 【${target}】 |\n\n- 【${target}】`
assert.equal(links(parse(mixed)).length, 3)
assert.ok(links(parse(mixed)).every((link) => link.url === target), 'normalization applies inside emphasis, tables and lists')
const positions = parse(observedAnswer).children[0] as Parent
const link = links(parse(observedAnswer))[0]
assert.equal(observedAnswer.slice(link.position!.start.offset!, link.position!.end.offset!), target, 'source offsets follow the corrected link extent')
assert.equal(positions.children.at(-1)?.type, 'text')
assert.equal((positions.children.at(-1) as { value: string }).value, '】')

// Exercise the real clickable component with the same sanitized href that the
// ReactMarkdown renderer passes to it; the main-process scheme gate is unchanged.
const opened: string[] = []
Object.defineProperty(globalThis, 'window', { configurable: true, value: { tensorfoldStudio: { openExternal: async (url: string) => { opened.push(url) } }, dispatchEvent: () => true } })
const { MarkdownLink } = await import('./markdown-link')
let clicked: ReturnType<typeof MarkdownLink> | null = null
renderToStaticMarkup(React.createElement(ReactMarkdown, {
  remarkPlugins: [remarkGfm, remarkCitationAutolinks],
  components: { a: (props) => { clicked = MarkdownLink(props); return clicked } },
}, observedAnswer))
assert.ok(clicked)
let prevented = false
;(clicked as ReturnType<typeof MarkdownLink>).props.onClick({ preventDefault: () => { prevented = true } })
assert.equal(prevented, true)
assert.deepEqual(opened, [target], 'clicking the model citation opens the exact usable URL through the existing bridge')

console.log('Markdown citation AST, actual renderer, preserved syntax and clickable URL regression checks passed.')
