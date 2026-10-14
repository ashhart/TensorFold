import { Icon } from '../../lib/icons'
import type { BuiltInCapability } from '../../lib/capabilities'

export function CapabilityGlyph({ capability }: { capability: BuiltInCapability }) {
  return (
    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-white shadow-[inset_0_1px_rgba(255,255,255,.18)]" style={{ background: capability.color }}>
      <Icon name={capability.icon} size={13} strokeWidth={1.9} />
    </span>
  )
}
