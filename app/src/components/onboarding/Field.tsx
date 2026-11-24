import type { ReactNode } from 'react'

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[13px] font-medium text-ink-2">{label}</span>
        {hint && <span className="text-[12px] text-ink-4">{hint}</span>}
      </div>
      {children}
    </label>
  )
}
