import React from 'react'
import { CodeBlock } from '../components/reading/CodeBlock'
import { DiagramBlock } from '../components/reading/DiagramBlock'
import { SafeMath } from '../components/reading/SafeMath'
import { MarkdownLink } from './markdown-link'

// Tiny dependency-free tokenizer. Good enough for colored code
// across JS/TS/JSON/shell without pulling in a multi-MB highlighter.
const KEYWORDS = new Set([
  'const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
  'break', 'continue', 'new', 'class', 'extends', 'implements', 'import', 'from', 'export', 'default',
  'async', 'await', 'yield', 'try', 'catch', 'finally', 'throw', 'typeof', 'instanceof', 'in', 'of',
  'type', 'interface', 'enum', 'namespace', 'public', 'private', 'protected', 'readonly', 'static',
  'true', 'false', 'null', 'undefined', 'this', 'super', 'void', 'as', 'def', 'fn', 'pub', 'use', 'struct',
])

type Tok = { t: string; c: string }

export function tokenize(code: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  const n = code.length
  const push = (t: string, c: string) => out.push({ t, c })

  while (i < n) {
    const ch = code[i]

    // line comment
    if ((ch === '/' && code[i + 1] === '/') || ch === '#') {
      let j = i
      while (j < n && code[j] !== '\n') j++
      push(code.slice(i, j), 'com'); i = j; continue
    }
    // block comment
    if (ch === '/' && code[i + 1] === '*') {
      let j = i + 2
      while (j < n && !(code[j] === '*' && code[j + 1] === '/')) j++
      j = Math.min(n, j + 2)
      push(code.slice(i, j), 'com'); i = j; continue
    }
    // string
    if (ch === '"' || ch === "'" || ch === '`') {
      let j = i + 1
      while (j < n && code[j] !== ch) { if (code[j] === '\\') j++; j++ }
      j = Math.min(n, j + 1)
      push(code.slice(i, j), 'str'); i = j; continue
    }
    // number
    if (/[0-9]/.test(ch)) {
      let j = i
      while (j < n && /[0-9a-fA-FxX._]/.test(code[j])) j++
      push(code.slice(i, j), 'num'); i = j; continue
    }
    // identifier / keyword / type
    if (/[A-Za-z_$]/.test(ch)) {
      let j = i
      while (j < n && /[A-Za-z0-9_$]/.test(code[j])) j++
      const word = code.slice(i, j)
      const cls = KEYWORDS.has(word) ? 'kw' : /^[A-Z]/.test(word) ? 'type' : code[j] === '(' ? 'fn' : 'id'
      push(word, cls); i = j; continue
    }
    // whitespace / punctuation
    let j = i
    while (j < n && !/[A-Za-z0-9_$"'`#]/.test(code[j]) && !(code[j] === '/' && (code[j + 1] === '/' || code[j + 1] === '*'))) j++
    if (j === i) j++
    push(code.slice(i, j), 'punc'); i = j
  }
  return out
}

export function highlight(code: string, _lang?: string): React.ReactNode[] {
  return tokenize(code).map((tok, k) =>
    tok.c === 'id' || tok.c === 'punc'
      ? <React.Fragment key={k}>{tok.t}</React.Fragment>
      : <span key={k} className={`syn-${tok.c}`}>{tok.t}</span>,
  )
}

function textFromNode(node: React.ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textFromNode).join('')
  if (React.isValidElement(node)) return textFromNode((node.props as any).children)
  return ''
}

// Keep Markdown links and citation repair owned by their existing safe handlers.
export const mdComponents = {
  a: MarkdownLink,
  pre({ children }: any) {
    const text = textFromNode(children).replace(/\n$/, '')
    const child = React.Children.toArray(children).find((node) => React.isValidElement(node)) as React.ReactElement<any> | undefined
    const language = /language-([^\s]+)/.exec(child?.props.className || '')?.[1]
    if (language === 'diagram' || language === 'mermaid') return <DiagramBlock source={text} />
    if (language === 'math') return <SafeMath expression={text} display />
    return <CodeBlock code={text} language={language} />
  },
  code({ children, className }: any) {
    return <code className={className}>{children}</code>
  },
  span({ children, ...props }: any) {
    if (typeof props['data-tf-math'] === 'string') return <SafeMath expression={props['data-tf-math']} display={props['data-tf-display'] === 'true'} />
    return <span>{children}</span>
  },
}
