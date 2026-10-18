import type { TelegramRemoteScope, TelegramRemoteTarget, TelegramRemoteTurnResult } from '../../../shared/telegram-remote'

export type TelegramRemoteCommand =
  | { kind: 'help' | 'chief' | 'rooms' | 'chats' | 'back' | 'unsupported' }
  | { kind: 'select'; target: Exclude<TelegramRemoteTarget, { kind: 'chief' }> }
  | { kind: 'message'; text: string }

export function parseTelegramRemoteCommand(text: string): TelegramRemoteCommand {
  const value = text.trim().replace(/^\/([a-z_]+)@[A-Za-z0-9_]+(?=\s|$)/i, '/$1')
  const simple = value.match(/^\/(help|start|chief|rooms|chats|back|leave)$/i)
  if (simple) {
    const kind = simple[1].toLowerCase()
    return { kind: kind === 'start' ? 'help' : kind === 'leave' ? 'back' : kind } as TelegramRemoteCommand
  }
  const select = value.match(/^\/(chat|room|enter)\s+([A-Za-z0-9_.:-]{1,160})$/i)
  if (select) return { kind: 'select', target: { kind: select[1].toLowerCase() === 'chat' ? 'chat' : 'room', id: select[2] } }
  return value.startsWith('/') ? { kind: 'unsupported' } : { kind: 'message', text: value }
}

export const TELEGRAM_REMOTE_HELP = [
  'TensorFold remote assistant',
  '/chief: chief of staff with your enabled chat and room scope',
  '/chats, /chat <id>: list and enter an allowed chat',
  '/rooms, /room <id>: list and enter an allowed room',
  '/back: return to chief of staff',
  'In chief mode: dispatch chat <id> <message> sends an explicit task to an allowed chat.',
  'Send a normal message after choosing a destination. New tasks keep the selected destination.',
  'Changing destination or access interrupts chat work started by this remote assistant.',
  'Tool and connector approvals stay in TensorFold on your Mac; Telegram cannot approve them.',
].join('\n')

export function listTelegramTargets(kind: 'rooms' | 'chats', scope: TelegramRemoteScope): string {
  const items = kind === 'rooms' ? scope.rooms : scope.threads
  if (!items.length) return `No allowed ${kind}. Choose the ${kind === 'rooms' ? 'rooms' : 'chats'} to expose in TensorFold settings.`
  const lines = items.slice(0, 20).map((item) => {
    const label = ('title' in item ? item.title : item.name).replace(/[\r\n]/g, ' ').slice(0, 100)
    return `${label} · ${item.id}`
  })
  if (items.length > 20) lines.push(`${items.length - 20} more allowed ${kind}; use an exact ID from TensorFold.`)
  return [`Allowed ${kind}:`, ...lines].join('\n')
}

export function telegramTurnReply(result: TelegramRemoteTurnResult): string {
  const body = String(result.text ?? '').trim()
  const run = result.runId ? ` Run ${result.runId.slice(0, 100)}.` : ''
  if (result.status === 'completed') return body || 'The run completed without a final text reply; open the selected chat in TensorFold for its results.'
  if (result.status === 'queued') return `${body || 'Queued in the selected destination.'}${run} Open TensorFold for progress and any required approvals.`
  if (result.status === 'busy') return body || 'The selected model or chat is busy; no message was sent. Try again when it is idle.'
  if (result.status === 'incomplete') return `${body || 'The run is incomplete.'}${run} Open TensorFold to continue or review approvals.`
  return body || 'TensorFold could not complete this request. Open the app for the run details.'
}
