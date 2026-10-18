import { authoredText, type UiTextParameters } from './language-packs'
import { currentUiLanguage } from './current-locale'
import { flowTextFor } from './flow-catalogue'
type Text = (source: string, params?: UiTextParameters) => string
const statusLabels: Record<string, string> = {
  idle: 'Idle', pending: 'Pending', queued: 'Queued', starting: 'Starting', loading: 'Loading', running: 'Running',
  streaming: 'Streaming', stopping: 'Stopping', stopped: 'Stopped', ready: 'Ready', active: 'Active', open: 'Open',
  complete: 'Complete', completed: 'Completed', done: 'Done', failed: 'Failed', error: 'Error', cancelled: 'Cancelled',
  canceled: 'Cancelled', blocked: 'Blocked', paused: 'Paused', draft: 'Draft', archived: 'Archived', resolved: 'Resolved',
  installed: 'Installed', downloaded: 'Downloaded', approved: 'Approved', denied: 'Denied', thinking: 'Thinking', tool: 'Working', partial: 'Partial', success: 'Success', cancelling: 'Stopping', cancel_requested: 'Stop requested', acknowledged: 'Acknowledged', in_progress: 'In progress',
  unknown: 'Unknown', unsupported: 'Unsupported', needs_review: 'Needs review', awaiting: 'Awaiting approval',
}
/** Presentation only: values used by routes, policies, sorting and controls stay unchanged. */
export function statusText(value: string | null | undefined, text: Text): string {
  if (!value) return ''
  const source = statusLabels[value], translated = source ? text(source) : value
  return translated === source ? value : translated
}
export function modelFitReason(reason: string | null | undefined, text: Text): string {
  if (!reason) return ''
  if (currentUiLanguage() === 'en') return reason
  const prefix = 'This model has not passed the required ', suffix = ' check; choose another model or run a new explicit check.'
  if (reason.startsWith(prefix) && reason.endsWith(suffix)) {
    const names: Record<string, string> = { text: 'Text', vision: 'Images', tools: 'Tool calls', reasoning: 'Reasoning channel' }
    const codes = reason.slice(prefix.length, -suffix.length).split(', ')
    if (codes.every(code => Object.hasOwn(names, code))) return text('This model has not passed the required {checks} check; choose another model or run a new explicit check.', { checks: codes.map(code => text(names[code])).join(', ') })
  }
  return text(reason)
}
export function modelCapabilityLabel(value: string, text: Text): string {
  if (currentUiLanguage() === 'en') return value
  const match = /^(Text|Images|Tools|Reasoning): (unknown|requested|observed|failed|unsupported|cancelled)( · required)?$/.exec(value)
  if (!match) return value
  const flow = flowTextFor(currentUiLanguage()), capability = match[1] === 'Tools' ? 'Tool calls' : match[1] === 'Reasoning' ? 'Reasoning channel' : match[1]
  return `${text(capability)}: ${flow.t(`qualification.${match[2]}`)}${match[3] ? ` · ${text('Required')}` : ''}`
}
/** Model/account names stay verbatim, even if a user names one Settings or Cancel. */
export function modelChoiceDetail(value: string, text: Text): string {
  if (currentUiLanguage() === 'en') return value
  for (const prefix of ['On this Mac', 'Planning only']) if (value.startsWith(prefix + ' · ')) return text(prefix) + value.slice(prefix.length)
  const parts = value.split(' · ')
  if (parts.length > 0 && parts.every(part => /(?:^(?:mlx|gguf)$| (?:context|memory)$)/i.test(part))) return parts.map(part => part.endsWith(' context') ? text('{value} context', { value: part.slice(0, -8) }) : part.endsWith(' memory') ? text('{value} memory', { value: part.slice(0, -7) }) : part).join(' · ')
  return text(value)
}
export function currentAuthoredText(source: string, params?: UiTextParameters): string { return authoredText(currentUiLanguage(), source, params) }
