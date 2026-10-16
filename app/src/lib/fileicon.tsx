import clsx from 'clsx'

// Small language-colored file badge. Functional color — like
// diff green/red — used only to identify file types, not as a brand hue.
const EXT: Record<string, { label: string; cls: string }> = {
  js: { label: 'JS', cls: 'bg-[#d9b23a]/22 text-[#e6c456]' },
  jsx: { label: 'JS', cls: 'bg-[#d9b23a]/22 text-[#e6c456]' },
  mjs: { label: 'JS', cls: 'bg-[#d9b23a]/22 text-[#e6c456]' },
  ts: { label: 'TS', cls: 'bg-[#3a86d9]/22 text-[#5aa6f0]' },
  tsx: { label: 'TS', cls: 'bg-[#3a86d9]/22 text-[#5aa6f0]' },
  css: { label: 'CSS', cls: 'bg-[#4a9de0]/18 text-[#74bdf5]' },
  scss: { label: 'SC', cls: 'bg-[#cf5a93]/20 text-[#e07ab0]' },
  html: { label: '<>', cls: 'bg-[#d96a35]/22 text-[#ef8a52]' },
  json: { label: '{}', cls: 'bg-[#d9b23a]/16 text-[#e6c456]' },
  md: { label: 'MD', cls: 'bg-ink-5 text-ink-2' },
  py: { label: 'PY', cls: 'bg-[#3a86d9]/20 text-[#5aa6f0]' },
  rs: { label: 'RS', cls: 'bg-[#c08a4a]/20 text-[#d9a766]' },
  go: { label: 'GO', cls: 'bg-[#4a9de0]/18 text-[#74bdf5]' },
  sh: { label: 'SH', cls: 'bg-[#5a9e5a]/20 text-[#7cc07c]' },
}

export function FileGlyph({ path, size = 16 }: { path: string; size?: number }) {
  const ext = path.split('.').pop()?.toLowerCase() || ''
  const def = EXT[ext] || { label: (ext.slice(0, 2) || '··').toUpperCase(), cls: 'bg-ink-5 text-ink-3' }
  return (
    <span
      style={{ width: size, height: size }}
      className={clsx('inline-flex shrink-0 items-center justify-center rounded font-bold leading-none', def.cls, size <= 16 ? 'text-[7.5px]' : 'text-[11px]')}
    >
      {def.label}
    </span>
  )
}
