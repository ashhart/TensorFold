import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
// Project-local skills with trust prompts (VST-010): repo-authored SKILL.md folders under
// `.vontra/skills/` are listed with their review state. Untrusted and changed-since-review
// skills never reach a run until the user trusts the CURRENT content — repo content is not
// implicitly trusted.
import { useEffect, useState } from 'react'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import type { ProjectSkillView } from '../lib/types'

const TRUST_TINT: Record<ProjectSkillView['trust'], string> = {
  trusted: 'text-positive',
  untrusted: 'text-ink-4',
  changed: 'text-warn',
}
const TRUST_LABEL: Record<ProjectSkillView['trust'], string> = {
  get trusted() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "trusted") },
  get untrusted() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "untrusted") },
  get changed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "changed since review") },
}

export function ProjectSkills({ projectId }: { projectId: string }) {
  const tx = useAuthoredText()
  const [views, setViews] = useState<ProjectSkillView[]>([])
  const [busy, setBusy] = useState<string | null>(null)

  const reload = () => { api.getProjectSkills(projectId).then(setViews).catch(() => setViews([])) }
  useEffect(reload, [projectId])

  const setTrust = async (view: ProjectSkillView, trusted: boolean) => {
    const bare = view.skill.id.replace(`ps_${projectId}_`, '')
    if (trusted && !confirm(tx("Trust \"{value0}\"?\n\nThis skill lives in the repo — anyone who can commit can change it. Trusting it lets agents in THIS project follow its instructions until the content changes again.", { value0: view.skill.name }))) return
    setBusy(view.skill.id)
    try {
      const next = await api.trustProjectSkill({ projectId, skillId: bare, trusted })
      setViews(next)
    } catch { reload() }
    finally { setBusy(null) }
  }

  if (views.length === 0) return null

  return (
    <div className="mt-4">
      <div className="mb-1.5 text-[13px] font-medium text-ink-2">
        {tx("Project skills ")}<span className="text-ink-4">{tx("· from .vontra/skills — repo content, trusted per version")}</span>
      </div>
      <div className="space-y-1">
        {views.map((view) => (
          <div key={view.skill.id} className="flex items-start gap-2 rounded-lg border border-line bg-card px-2.5 py-1.5">
            <Icon name={view.trust === 'trusted' ? 'check' : view.trust === 'changed' ? 'warning' : 'shield'} size={13} className={`mt-0.5 shrink-0 ${TRUST_TINT[view.trust]}`} />
            <span className="min-w-0 flex-1">
              <span className="block text-[13px] font-medium text-ink">{view.skill.name}</span>
              {view.skill.description && <span className="block truncate text-[13px] text-ink-3">{view.skill.description}</span>}
              <span className={`block text-[12px] ${TRUST_TINT[view.trust]}`}>
                {TRUST_LABEL[view.trust]}
                {view.trust === 'changed' ? tx(" — the file was edited after you trusted it; it stays out of runs until re-trusted") : ''}
                {view.trust === 'untrusted' ? tx(" — not used by runs") : ''}
              </span>
            </span>
            {view.trust === 'trusted' ? (
              <button
                onClick={() => void setTrust(view, false)}
                disabled={busy === view.skill.id}
                className="shrink-0 rounded-md border border-line px-2 py-1 text-[12px] text-ink-3 hover:text-danger disabled:opacity-45"
              >
                {tx("Revoke")}</button>
            ) : (
              <button
                onClick={() => void setTrust(view, true)}
                disabled={busy === view.skill.id}
                className="shrink-0 rounded-md bg-positive/15 px-2 py-1 text-[12px] font-medium text-positive hover:bg-positive/25 disabled:opacity-45"
              >
                {view.trust === 'changed' ? tx("Re-trust") : tx("Trust")}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
