import { Icon } from '../../lib/icons'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Menu } from '../ui'

export function ConversationTools({ helpersDisabled, exportDisabled, onHelpers, onExport }: {
  helpersDisabled: boolean
  exportDisabled: boolean
  onHelpers: () => void
  onExport: () => void
}) {
  const tx = useAuthoredText()
  const item = 'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] text-ink-2 hover:bg-ink-5 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent'
  return <Menu width={264} align="end" placement="bottom" trigger={open => <button type="button"
    aria-label={tx('More conversation tools')} aria-expanded={open}
    className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-2 hover:bg-ink-5 focus-visible:outline-2 focus-visible:outline-accent">
    <Icon name="dots" size={16} />
  </button>}>
    {close => <div>
      <p className="px-3 pb-1 pt-2 text-[11px] font-medium text-ink-3">{tx('Conversation tools')}</p>
      <button type="button" disabled={helpersDisabled} className={item}
        title={helpersDisabled ? tx('Local helpers require an idle, local conversation.') : undefined}
        onClick={() => { close(); onHelpers() }}><Icon name="cpu" size={15} />{tx('Local helpers')}</button>
      <button type="button" disabled={exportDisabled} className={item}
        title={exportDisabled ? tx('Readable export is available on the conversation’s local host.') : undefined}
        onClick={() => { close(); onExport() }}><Icon name="fileText" size={15} />{tx('Export readable text')}</button>
    </div>}
  </Menu>
}
