import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { Icon } from '../../lib/icons'

export const TASK_STARTERS = [
  { id: 'summarise', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Summarise a file") }, icon: 'fileText', prompt: 'Summarise the file I attach. Highlight the main points, decisions and next steps. If any contents cannot be read, tell me what is missing before making conclusions.', attachFile: true },
  { id: 'code', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Write or explain code") }, icon: 'code', prompt: 'Help me write or understand some code. First ask what I want to build or explain and which language or project I am working with.', attachFile: false },
  { id: 'plan', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Plan a task") }, icon: 'list', prompt: 'Help me turn an idea into a practical plan. Ask me about the goal, any constraints and what a successful result would look like before suggesting the steps.', attachFile: false },
] as const

export type TaskStarterId = typeof TASK_STARTERS[number]['id']

/** Starters prepare an editable draft; sending and attachment review remain explicit. */
export function TaskStarters({ onSelect }: { onSelect: (id: TaskStarterId) => void }) {
  const tx = useAuthoredText()
  return <div role="group" aria-label={tx("Start with a task")} className="mb-3 flex flex-wrap justify-center gap-2">
    {TASK_STARTERS.map((starter) => <button key={starter.id} type="button" onClick={() => onSelect(starter.id)} className="inline-flex min-h-9 items-center gap-2 rounded-full border border-line bg-card px-3 py-2 text-[12px] text-ink-2 transition-colors hover:border-line-strong hover:bg-ink-5 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"><Icon name={starter.icon} size={14} />{starter.label}</button>)}
  </div>
}
