import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import type { Settings as TSettings } from '../../lib/types'
import { Field } from '../ui'
import { Section } from './shared'

const parseTerms = (text: string) => [...new Set(text.split('\n').map((line) => line.trim()).filter(Boolean))].slice(0, 200)
const fieldClass = 'w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink placeholder:text-ink-4 focus:border-ink-3 focus:outline-none'

// Planning-only guard settings: names to withhold from external models, and a test box that
// shows what an external model would receive (the host runs the real redaction).
export function ConfidentialTermsSection({ draft, patch }: { draft: TSettings; patch: (p: Partial<TSettings>) => void }) {
  const saved = (draft.confidentialTerms ?? []).join('\n')
  const [termsText, setTermsText] = useState(saved)
  const [sample, setSample] = useState('')
  const [preview, setPreview] = useState<{ text: string; withheld: string; total: number } | null>(null)
  useEffect(() => { setTermsText(saved) }, [saved])

  const terms = parseTerms(termsText)
  const termsKey = terms.join('\n')
  useEffect(() => {
    if (!sample.trim()) { setPreview(null); return }
    let live = true
    const timer = setTimeout(() => {
      api.previewWithheld({ text: sample, terms }).then((r) => { if (live) setPreview(r) }).catch(() => { if (live) setPreview(null) })
    }, 250)
    return () => { live = false; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sample, termsKey])

  const commit = () => { if (termsKey !== saved) patch({ confidentialTerms: terms }) }

  return (
    <Section
      title="Confidential terms"
      desc="Cloud providers and the Codex, Claude and other CLI engines only plan. Before anything is sent, tool results, files and images are withheld, and secrets, personal details (emails, phone numbers, IP addresses, card numbers, IBANs, SSNs, NI numbers, your username, hostname and home folder), the workspace path and these terms become placeholders. Replies are filled back in on this Mac."
    >
      <div className="space-y-3 rounded-xl border border-line bg-card p-3.5 shadow-card">
        <Field label="Terms to withhold" hint={`${terms.length} term${terms.length === 1 ? '' : 's'} · one per line, any case`}>
          <textarea
            value={termsText}
            onChange={(e) => setTermsText(e.target.value)}
            onBlur={commit}
            rows={4}
            placeholder={'Company name\nClient name\nProject codename'}
            className={fieldClass}
          />
        </Field>
        <Field label="Test" hint="Paste text to see what an external model would receive">
          <textarea value={sample} onChange={(e) => setSample(e.target.value)} rows={3} placeholder="e.g. Email the Acme launch plan to someone@example.com" className={fieldClass} />
        </Field>
        {preview && (
          <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
            <div className="mb-1 text-[13px] text-ink-3">{preview.total ? `${preview.total} item${preview.total === 1 ? '' : 's'} withheld: ${preview.withheld}` : 'Nothing would be withheld.'}</div>
            <pre className="whitespace-pre-wrap break-words font-mono text-[13px] text-ink-2">{preview.text}</pre>
          </div>
        )}
      </div>
    </Section>
  )
}
