import clsx from 'clsx'
import type { SshConnectionState } from '../../lib/types'

export function SshStatusDot({ phase }: { phase?: SshConnectionState['phase'] }) {
  const color = phase === 'connected' ? 'bg-positive'
    : phase === 'error' || phase === 'disconnected' ? 'bg-danger'
    : phase ? 'bg-warn' : 'bg-ink-4'
  return <span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', color)} />
}
