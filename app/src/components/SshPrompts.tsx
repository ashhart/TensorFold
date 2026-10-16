import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'

// Modal for ssh2's interactive prompts: key passphrase / password, and unknown
// or changed host-key trust decisions. Secrets are held only in renderer memory
// for the lifetime of the prompt and sent straight to the connection — never saved.
export function SshPrompts() {
  const secret = useStore((s) => s.sshSecretPrompt)
  const trust = useStore((s) => s.sshTrustPrompt)
  const resolveSecret = useStore((s) => s.resolveSshSecret)
  const resolveTrust = useStore((s) => s.resolveSshTrust)
  const [value, setValue] = useState('')

  useEffect(() => { setValue('') }, [secret?.promptId])

  if (!secret && !trust) return null

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="w-[400px] rounded-2xl border border-line bg-card p-5 shadow-2xl rise">
        {secret ? (
          <form onSubmit={(e) => { e.preventDefault(); resolveSecret(value); setValue('') }}>
            <div className="mb-1 flex items-center gap-2 text-[13px] font-semibold text-ink">
              <Icon name="lock" size={15} className="text-ink-3" />
              SSH {secret.kind === 'passphrase' ? 'key passphrase' : 'password'}
            </div>
            <div className="mb-3 text-[13px] leading-snug text-ink-3">
              {secret.label}{secret.retry ? ' · previous attempt failed' : ''}
            </div>
            <input
              autoFocus
              type="password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={secret.kind === 'passphrase' ? 'Key passphrase' : 'Password'}
              className="w-full rounded-lg border border-line bg-canvas px-3 py-2 text-[13px] text-ink outline-none focus:border-line-strong"
            />
            <div className="mt-1.5 text-[12px] text-ink-4">Held only in memory for this connection — never saved.</div>
            <div className="mt-4 flex justify-end gap-2">
              <button type="button" onClick={() => { resolveSecret(null); setValue('') }} className="rounded-lg px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5">Cancel</button>
              <button type="submit" className="rounded-lg bg-accent px-3 py-1.5 text-[13px] font-semibold text-accent-ink transition-colors hover:bg-accent-strong">Continue</button>
            </div>
          </form>
        ) : trust ? (
          <div>
            <div className={clsx('mb-1 flex items-center gap-2 text-[13px] font-semibold', trust.changed ? 'text-danger' : 'text-ink')}>
              <Icon name="shield" size={15} className={trust.changed ? 'text-danger' : 'text-ink-3'} />
              {trust.changed ? 'Host key changed' : 'Unknown host key'}
            </div>
            <div className="mb-3 text-[13px] leading-snug text-ink-3">
              {trust.changed
                ? <>The server key for <span className="text-ink-2">{trust.host}:{trust.port}</span> differs from the one previously trusted. This can mean the server changed — or a man-in-the-middle attack.</>
                : <>First connection to <span className="text-ink-2">{trust.host}:{trust.port}</span>. Verify the fingerprint out-of-band before trusting.</>}
            </div>
            <div className="rounded-lg border border-line bg-canvas px-3 py-2 font-mono text-[12px] break-all text-ink-2">{trust.keyType} SHA256:{trust.fingerprintSha256}</div>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => resolveTrust(false)} className="rounded-lg px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5">Cancel</button>
              <button onClick={() => resolveTrust(true)} className={clsx('rounded-lg px-3 py-1.5 text-[13px] font-semibold transition-colors', trust.changed ? 'bg-danger text-bg hover:opacity-90' : 'bg-accent text-accent-ink hover:bg-accent-strong')}>
                {trust.changed ? 'Trust anyway' : 'Trust & continue'}
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
