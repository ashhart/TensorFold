import clsx from 'clsx'
import { Icon } from '../../../lib/icons'
import type { ConnectorStatus } from '../../../lib/connector-status'

// Transient confirmation that an add/save/sync actually happened — fixes the
// "I clicked Sync and nothing happened" gap (a successful no-op had no feedback).
export function ConnectorStatusChip({ status }: { status?: ConnectorStatus }) {
  if (!status) return null
  return (
    <div className={clsx(
      'mt-1.5 inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[12px]',
      status.state === 'ok' ? 'border-positive/30 bg-positive/10 text-positive' : 'border-danger/30 bg-danger/5 text-danger',
    )}>
      <Icon name={status.state === 'ok' ? 'check' : 'warning'} size={12} />
      <span className="truncate">{status.msg}</span>
    </div>
  )
}
