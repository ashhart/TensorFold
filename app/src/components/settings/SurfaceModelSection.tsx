import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { modelDisplayName, providerAccountChoices } from '../../lib/provider-accounts'
import type { Settings } from '../../lib/types'
import { SELECT_CLASS, Section } from './shared'

export function SurfaceModelSection({ draft, patch, title, desc, providerKey, accountKey, modelKey }: {
  draft: Settings
  patch: (p: Partial<Settings>) => void
  title: string
  desc: string
  providerKey: 'assistantProviderId' | 'codeProviderId'
  accountKey: 'assistantProviderAccountId' | 'codeProviderAccountId'
  modelKey: 'assistantModel' | 'codeModel'
}) {
  const tx = useAuthoredText()
  const providerId = draft[providerKey] ?? null
  const provider = providerId ? draft.providers.find((p) => p.id === providerId) ?? null : null
  const accounts = provider ? providerAccountChoices(provider) : []
  const account = accounts.find((choice) => choice.id === (draft[accountKey] ?? null))
    ?? accounts.find((choice) => choice.id === (provider?.defaultAccountId ?? null))
    ?? accounts[0]
  const models = account?.models?.length ? account.models : (provider?.models ?? [])
  return (
    <Section title={title} desc={desc}>
      <div className="flex flex-wrap items-center gap-2">
        <select
          className={SELECT_CLASS}
          value={providerId ?? ''}
          onChange={(e) => {
            const pid = e.target.value
            if (!pid) { patch({ [providerKey]: null, [accountKey]: null, [modelKey]: null }); return }
            const prov = draft.providers.find((p) => p.id === pid)
            const nextAccounts = prov ? providerAccountChoices(prov) : []
            const nextAccount = nextAccounts.find((choice) => choice.id === (prov?.defaultAccountId ?? null)) ?? nextAccounts[0]
            const nextModels = nextAccount?.models?.length ? nextAccount.models : (prov?.models ?? [])
            patch({ [providerKey]: pid, [accountKey]: nextAccount?.id ?? null, [modelKey]: nextModels[0] ?? null })
          }}
        >
          <option value="">{tx("Same as active model")}</option>
          {draft.providers.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        {provider && (
          <>
            <select
              className={SELECT_CLASS}
              value={account?.id ?? ''}
              onChange={(e) => {
                const nextAccount = accounts.find((choice) => (choice.id ?? '') === e.target.value) ?? accounts[0]
                const nextModels = nextAccount?.models?.length ? nextAccount.models : provider.models
                patch({ [accountKey]: nextAccount?.id ?? null, [modelKey]: nextModels[0] ?? null })
              }}
            >
              {accounts.map((choice) => <option key={choice.id ?? 'default'} value={choice.id ?? ''}>{choice.name}</option>)}
            </select>
            <select className={SELECT_CLASS} value={draft[modelKey] ?? ''} onChange={(e) => patch({ [modelKey]: e.target.value || null })}>
              {models.length === 0 && <option value="">{tx("(no models — connect this provider first)")}</option>}
              {models.map((m) => <option key={m} value={m}>{modelDisplayName(provider, m)}</option>)}
            </select>
          </>
        )}
      </div>
    </Section>
  )
}
