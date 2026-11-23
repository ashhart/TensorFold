import { createContext, useContext, useEffect, useRef, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

type CopyProps = { message: number; onCopy: (text: string) => Promise<boolean> }
const CopyContext = createContext<CopyProps | null>(null)

const CodeBlock: NonNullable<Components['pre']> = ({ children, node }) => {
  const copy = useContext(CopyContext)!
  const [done, setDone] = useState(false)
  const reset = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const pre = useRef<HTMLPreElement>(null)
  useEffect(() => () => clearTimeout(reset.current), [])

  return (
    <pre ref={pre}>
      {children}
      <button type="button" className="btn sm copy"
        aria-label={`Copy code block at line ${node?.position?.start.line ?? 1} in message ${copy.message}`}
        onClick={async event => {
          event.currentTarget.focus({ preventScroll: true })
          const code = pre.current?.querySelector('code')
          if (!code) return
          const copied = await copy.onCopy(code.textContent ?? '')
          if (!pre.current) return
          clearTimeout(reset.current)
          setDone(copied)
          if (copied) reset.current = setTimeout(() => setDone(false), 1200)
        }}>{done ? 'Copied' : 'Copy'}</button>
    </pre>
  )
}

// Keep renderer identities stable when the transcript or its parent updates.
const components: Components = {
  pre: CodeBlock,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
  // Link to images so rendering a reply can't send conversation data to the image host.
  img: ({ src, alt }) => <a href={typeof src === 'string' ? src : undefined} target="_blank" rel="noreferrer" title="Image not loaded: open it yourself">[image: {alt || 'no description'}]</a>,
}

export default function Markdown({ text, onCopy, message }: CopyProps & { text: string }) {
  return (
    <div className="md">
      <CopyContext.Provider value={{ onCopy, message }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{text}</ReactMarkdown>
      </CopyContext.Provider>
    </div>
  )
}
