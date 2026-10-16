import React from 'react'
import { api } from './ipc'
import { Icon } from './icons'
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

function CopyCodeButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false)
  const copy = async () => {
    await api.writeClipboard(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }
  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"
    >
      <Icon name={copied ? 'check' : 'copy'} size={12} />
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

// react-markdown component overrides (external links, highlighted code).
export const mdComponents = {
  // Model-produced links open externally; see markdown-link.ts.
  a: MarkdownLink,
  pre({ children, ...props }: any) {
    const text = textFromNode(children).replace(/\n$/, '')
    return (
      <div className="my-3 overflow-hidden rounded-lg border border-line bg-card-2">
        <pre {...props} className="scroll-thin overflow-x-auto p-3">{children}</pre>
        <div className="flex justify-end border-t border-line px-2 py-1">
          <CopyCodeButton text={text} />
        </div>
      </div>
    )
  },
  code({ className, children, ...props }: any) {
    const text = Array.isArray(children) ? children.join('') : String(children ?? '')
    const lang = /language-(\w+)/.exec(className || '')?.[1]
    const isBlock = text.includes('\n') || !!lang
    if (!isBlock) return <code {...props}>{children}</code>
    return <code className="syn-code">{highlight(text.replace(/\n$/, ''), lang)}</code>
  },
}
