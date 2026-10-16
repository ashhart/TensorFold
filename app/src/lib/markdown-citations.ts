import type { Link, Parent, Root, Text } from 'mdast'
import type { VFile } from 'vfile'

const CLOSERS = new Map([
  ['】', '【'], ['〗', '〖'], ['〕', '〔'], ['］', '［'],
  ['」', '「'], ['』', '『'], ['〉', '〈'], ['》', '《'], ['）', '（'],
])
const OPENERS = new Map([...CLOSERS].map(([close, open]) => [open, close]))

function citationSuffix(value: string): string {
  const balance = new Map([...CLOSERS.keys()].map((close) => [close, 0]))
  // Count an encoded opener too, so a valid mixed-encoding URL such as
  // /%E3%80%90section】 keeps its closing bracket. Never rewrite encoded bytes.
  const punctuation = value.replace(/(?:%[0-9a-f]{2})+/gi, (encoded) => {
    try { return decodeURIComponent(encoded) } catch { return encoded }
  })
  for (const char of punctuation) {
    const close = OPENERS.get(char)
    if (close) balance.set(close, balance.get(close)! + 1)
    else if (CLOSERS.has(char)) balance.set(char, balance.get(char)! - 1)
  }
  let end = value.length
  while (end > 0) {
    const char = value[end - 1]
    if (!CLOSERS.has(char) || balance.get(char)! >= 0) break
    balance.set(char, balance.get(char)! + 1)
    end--
  }
  return value.slice(end)
}

/** Run after GFM, using source spans to leave authored links and code untouched. */
export function remarkCitationAutolinks() {
  return (tree: Root, file: VFile): void => {
    const source = file.toString()
    const visit = (parent: Parent): void => {
      for (let index = 0; index < parent.children.length; index++) {
        const node = parent.children[index]
        if (node.type !== 'link') {
          if ('children' in node) visit(node as Parent)
          continue
        }
        const link = node as Link
        const start = link.position?.start.offset
        const end = link.position?.end.offset
        const label = link.children.length === 1 && link.children[0].type === 'text' ? link.children[0] : null
        if (start === undefined || end === undefined || !label) continue
        // Explicit [label](url), <url> and references have different source spans;
        // never repair their destinations based on what their labels look like.
        if (!/^https?:\/\//i.test(source.slice(start, end))) continue
        if (link.url !== label.value) continue
        const suffix = citationSuffix(label.value)
        if (!suffix || !link.url.endsWith(suffix)) continue
        const originalEnd = { ...link.position!.end }
        const trimmedEnd = { ...originalEnd, column: originalEnd.column - suffix.length, offset: end - suffix.length }
        link.url = link.url.slice(0, -suffix.length)
        label.value = label.value.slice(0, -suffix.length)
        link.position!.end = trimmedEnd
        if (label.position) label.position.end = { ...trimmedEnd }
        const punctuation: Text = { type: 'text', value: suffix, position: { start: { ...trimmedEnd }, end: originalEnd } }
        parent.children.splice(index + 1, 0, punctuation)
        index++
      }
    }
    visit(tree)
  }
}
