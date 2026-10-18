import { useAuthoredText } from '../lib/preferences/useAuthoredText'

/** Host, provider and file errors remain data even when they resemble an interface label. */
export function OriginalDiagnostic({ children: value }: { children: string | number | null | undefined }) {
  const tx = useAuthoredText()
  if (value == null || value === '') return null
  return <span className="block min-w-0 break-words" data-original-diagnostic>
    <span className="block text-[11px] font-medium opacity-75">{tx('Original diagnostic')}</span>
    <span dir="auto" className="select-text whitespace-pre-wrap">{value}</span>
  </span>
}
