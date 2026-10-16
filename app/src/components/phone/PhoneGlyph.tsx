export function PhoneGlyph({ size = 17, className = '' }: { size?: number; className?: string }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}><rect x="6.5" y="2.5" width="11" height="19" rx="2.5" /><path d="M10.5 5h3M11.5 18.5h1" /></svg>
}
