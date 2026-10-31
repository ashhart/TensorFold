import { useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/** A code block with its own Copy button; the text is taken from the rendered children. */
function Pre({ children, onCopy }: { children?: ReactNode; onCopy: (text: string) => void }) {
  const [done, setDone] = useState(false)
  const text = (node: ReactNode): string => {
    if (node == null || typeof node === 'boolean') return ''
    if (typeof node === 'string' || typeof node === 'number') return String(node)
    if (Array.isArray(node)) return node.map(text).join('')
    if (typeof node === 'object' && 'props' in node) return text((node as { props: { children?: ReactNode } }).props.children)
    return ''
  }
  return (
    <pre>
      {children}
      <button type="button" className="btn sm copy" onClick={() => { onCopy(text(children)); setDone(true); setTimeout(() => setDone(false), 1200) }}>{done ? 'Copied' : 'Copy'}</button>
    </pre>
  )
}

/** The model's reply as Markdown (GitHub flavour, no raw HTML, no remote images), with copyable code blocks. */
export default function Markdown({ text, onCopy }: { text: string; onCopy: (text: string) => void }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ pre: ({ children }) => <Pre onCopy={onCopy}>{children}</Pre>, a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
        // an image in a reply is not fetched: a model could put the conversation in its address and send it away
        img: ({ src, alt }) => <a href={typeof src === 'string' ? src : undefined} target="_blank" rel="noreferrer" title="Image not loaded: open it yourself">[image: {alt || 'no description'}]</a> }}>
        {text}
      </ReactMarkdown>
    </div>
  )
}
