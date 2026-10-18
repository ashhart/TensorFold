import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { usePhone } from '../../lib/phone'
import { PhoneGlyph } from './PhoneGlyph'
import { phoneChatGranted } from './phone-format'

export interface PhoneButtonProps { threadId?: string | null; open: boolean; onClick: () => void; disabled?: boolean }
export function PhoneButton({ threadId, open, onClick, disabled }: PhoneButtonProps) {
  const tx = useAuthoredText()
  const { snapshot } = usePhone()
  const granted = phoneChatGranted(snapshot?.session, threadId)
  return <button type="button" aria-label={granted ? tx("Phone control is enabled for this chat") : tx("Connect a phone")} aria-expanded={open} aria-controls="phone-chat-panel" title={granted ? tx("Phone control is enabled for this chat") : tx("Connect a phone with USB or Wi-Fi")} disabled={disabled} onClick={onClick} className={clsx('inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2 text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40', open ? 'bg-ink-5 text-ink' : granted ? 'text-accent hover:bg-ink-5' : 'text-ink-2 hover:bg-ink-5')}><PhoneGlyph size={16} /><span className="hidden @sm:inline">{tx("Phone")}</span>{granted && <span className="h-1.5 w-1.5 rounded-full bg-positive" />}</button>
}
