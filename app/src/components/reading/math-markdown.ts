// An intentionally small remark extension. Code, authored HTML and link targets
// remain owned by their existing Markdown handlers; only text leaves are split.
export function remarkBoundedMath() {
  return (tree: any, file: { toString(): string }) => {
    let expressions = 0
    const source = file.toString()
    const visit = (parent: any): void => {
      if (!Array.isArray(parent.children) || ['code', 'inlineCode', 'html', 'tfMath', 'link', 'linkReference'].includes(parent.type)) return
      const children: any[] = []
      for (const node of parent.children) {
        if (node.type !== 'text' || expressions >= 80 || typeof node.value !== 'string') { visit(node); children.push(node); continue }
        const raw = source.slice(node.position?.start?.offset ?? 0, node.position?.end?.offset ?? 0)
        if (raw.includes('\\$') || node.value.length > 40_000) { children.push(node); continue }
        const text = node.value
        const pattern = /\$\$([^$]{1,4096})\$\$|(?<!\$)\$(?!\$)([^$\n]{1,2048})\$(?!\$)/g
        let start = 0, match: RegExpExecArray | null
        while (expressions < 80 && (match = pattern.exec(text))) {
          const expression = match[1] ?? match[2]
          // Currency prose is left readable; single-symbol formulas still work.
          if (!match[1] && /^\d/.test(expression) && /\s/.test(expression)) continue
          if (match.index > start) children.push({ type: 'text', value: text.slice(start, match.index) })
          children.push({ type: 'tfMath', value: expression, data: { hName: 'span', hProperties: { 'data-tf-math': expression, 'data-tf-display': match[1] !== undefined ? 'true' : 'false' } } })
          start = match.index + match[0].length; expressions++
        }
        if (start < text.length) children.push({ type: 'text', value: text.slice(start) })
      }
      parent.children = children
    }
    visit(tree)
  }
}
