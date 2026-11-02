import { createHash, randomUUID } from 'node:crypto'
import { createSerialTaskQueue } from '../serial-task-queue'
import type { EmailSourceConfig } from '../types'
import type { MicrosoftMailSignIn, MicrosoftMailPoll } from '../../../shared/microsoft-mail'
import type { MicrosoftMailAccount, MicrosoftMailState, MicrosoftMailDeps } from './types'
import { GRAPH, LOGIN, MicrosoftMailTransport } from './transport'

interface SignIn extends MicrosoftMailSignIn { clientId: string; deviceCode: string; nextPollAt: number; epoch: number; allowSend: boolean }

export function microsoftMailSource(account: Pick<MicrosoftMailAccount, 'id' | 'email' | 'revision'>, now = Date.now()): EmailSourceConfig {
  return {
    id: account.id, label: `Outlook · ${account.email}`, kind: 'microsoft_graph', path: `${GRAPH}/v1.0/me/messages`,
    user: account.email, microsoftRevision: account.revision, enabled: true, createdAt: now, updatedAt: now, lastSyncAt: null, lastError: null
  }
}

export class MicrosoftMailAccounts {
  constructor(private deps: MicrosoftMailDeps, private transport: MicrosoftMailTransport) { this.now = deps.now ?? Date.now }
  private state: MicrosoftMailState = { version: 1, accounts: [] }

  private loaded = false

  private queue = createSerialTaskQueue()

  private sessions = new Map<string, SignIn>()

  private epoch = 0

  private now: () => number

  private async load() {
    if (this.loaded) return
    const state = await this.deps.store.read()
    if (state && (state.version !== 1 || !Array.isArray(state.accounts))) throw new Error('Microsoft mail storage cannot be read. Restore the protected account store.')
    this.state = state ?? this.state
    this.loaded = true
  }

  private current(epoch: number) { if (epoch !== this.epoch) throw new Error('Microsoft connection changed. Sign in or sync again.') }

  private async commit(accounts: MicrosoftMailAccount[], epoch: number) {
    this.current(epoch)
    const next: MicrosoftMailState = { version: 1, accounts }
    await this.deps.store.write(structuredClone(next), () => this.current(epoch))
    this.current(epoch)
    this.state = next
  }

  private token(form: Record<string, string>) {
    return this.transport.json(LOGIN, '/common/oauth2/v2.0/token', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams(form).toString() })
  }

  private scopes(value: unknown): string[] {
    return typeof value === 'string' ? value.split(/\s+/).map(scope => scope.replace(/^https:\/\/graph.microsoft.com\//i, '').toLowerCase()).filter(Boolean) : []
  }

  private credentials(json: any, prior?: MicrosoftMailAccount): Pick<MicrosoftMailAccount, 'accessToken' | 'refreshToken' | 'expiresAt' | 'scopes'> {
    const scopes = json.scope ? this.scopes(json.scope) : prior?.scopes ?? []
    if (typeof json.access_token !== 'string' || !json.access_token || !Number.isFinite(json.expires_in) || json.expires_in <= 0 || !scopes.includes('mail.read')) throw new Error('Microsoft did not grant mail reading. Review the application permissions and reconnect.')
    const refreshToken = json.refresh_token ?? prior?.refreshToken
    if (typeof refreshToken !== 'string' || !refreshToken) throw new Error('Microsoft did not grant offline access. Reconnect the account.')
    return { accessToken: json.access_token, refreshToken, expiresAt: this.now() + Math.min(json.expires_in, 86_400) * 1000, scopes }
  }

  async begin(clientId: string, allowSend: boolean): Promise<MicrosoftMailSignIn> {
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(clientId)) throw new Error('Enter your Microsoft Entra public-client application ID.')
    if (typeof allowSend !== 'boolean') throw new Error('Choose the requested Microsoft mail permissions.')
    for (const [id, session] of this.sessions) if (session.expiresAt <= this.now()) this.sessions.delete(id)
    if (this.sessions.size >= 3) throw new Error('Finish or cancel the current Microsoft sign-in first.')
    const epoch = this.epoch
    const { response, json } = await this.transport.json(LOGIN, '/common/oauth2/v2.0/devicecode', {
      method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ client_id: clientId, scope: `offline_access User.Read Mail.Read${allowSend ? ' Mail.Send' : ''}` }).toString()
    })
    this.current(epoch)
    if (!response.ok || typeof json.device_code !== 'string' || typeof json.user_code !== 'string' || !Number.isFinite(json.expires_in) || json.expires_in <= 0) throw new Error('Microsoft sign-in could not start. Check the public-client application registration and tenant policy.')
    const publicState: MicrosoftMailSignIn = {
      id: randomUUID(), userCode: json.user_code, verificationUri: 'https://microsoft.com/devicelogin',
      expiresAt: this.now() + Math.min(json.expires_in, 900) * 1000, intervalMs: Math.max(5, Math.min(Number(json.interval) || 5, 60)) * 1000
    }
    this.sessions.set(publicState.id, { ...publicState, clientId, deviceCode: json.device_code, nextPollAt: this.now() + publicState.intervalMs, epoch, allowSend })
    return publicState
  }

  cancel(id: string): void { this.sessions.delete(id) }

  async connections() {
    return this.queue.run('state', async () => {
      await this.load()
      return this.state.accounts.map(account => ({ id: account.id, email: account.email, revision: account.revision, read: account.scopes.includes('mail.read'), send: account.scopes.includes('mail.send') }))
    })
  }

  async poll(id: string): Promise<MicrosoftMailPoll & { source?: EmailSourceConfig }> {
    return this.queue.run('state', async () => {
      const session = this.sessions.get(id)
      if (!session || session.expiresAt <= this.now()) { this.sessions.delete(id); throw new Error('Microsoft sign-in expired or was canceled. Start again.') }
      if (this.now() < session.nextPollAt) return { status: 'pending', intervalMs: session.intervalMs }
      session.nextPollAt = this.now() + session.intervalMs
      const { response, json } = await this.token({ client_id: session.clientId, device_code: session.deviceCode, grant_type: 'urn:ietf:params:oauth:grant-type:device_code' })
      this.current(session.epoch)
      if (this.sessions.get(id) !== session) throw new Error('Microsoft sign-in was canceled. Nothing was connected.')
      if (!response.ok) {
        if (json.error === 'authorization_pending' || json.error === 'slow_down') {
          if (json.error === 'slow_down') session.intervalMs = Math.min(60_000, session.intervalMs + 5000)
          session.nextPollAt = this.now() + session.intervalMs
          return { status: 'pending', intervalMs: session.intervalMs }
        }
        this.sessions.delete(id)
        throw new Error('Microsoft sign-in was declined, expired or blocked by tenant policy. Start again or contact your administrator.')
      }
      const credentials = this.credentials(json)
      credentials.scopes = credentials.scopes.filter(scope => scope !== 'mail.send' || session.allowSend)
      const profile = await this.transport.json(GRAPH, '/v1.0/me?$select=id,mail,userPrincipalName', { headers: { Authorization: `Bearer ${credentials.accessToken}` } })
      this.current(session.epoch)
      if (this.sessions.get(id) !== session) throw new Error('Microsoft sign-in was canceled. Nothing was connected.')
      const email = profile.json?.mail || profile.json?.userPrincipalName
      if (!profile.response.ok || typeof profile.json?.id !== 'string' || typeof email !== 'string' || !/^[^\s@<>;,]+@[^\s@<>;,]+$/.test(email)) throw new Error('Microsoft account identity could not be verified. Reconnect the account.')
      await this.load()
      const accountId = `ms-${createHash('sha256').update(`${session.clientId}:${profile.json.id}`).digest('hex').slice(0, 24)}`
      const previous = this.state.accounts.find(account => account.id === accountId)
      const account: MicrosoftMailAccount = { id: accountId, clientId: session.clientId, email, revision: (previous?.revision ?? 0) + 1, ...credentials }
      await this.commit([...this.state.accounts.filter(item => item.id !== accountId), account], session.epoch)
      this.sessions.delete(id)
      return { status: 'connected', source: microsoftMailSource(account, this.now()) }
    })
  }

  async remove(id: string): Promise<void> {
    const epoch = ++this.epoch
    this.sessions.clear()
    return this.queue.run('state', async () => { await this.load(); await this.commit(this.state.accounts.filter(account => account.id !== id), epoch) })
  }

  async withAccess<T>(source: EmailSourceConfig, action: (account: MicrosoftMailAccount, assertCurrent: () => void) => Promise<T>): Promise<T> {
    const epoch = this.epoch
    return this.queue.run('state', async () => {
      this.current(epoch)
      this.deps.assertSourceCurrent?.(source)
      await this.load()
      let account = this.state.accounts.find(item => item.id === source.id)
      if (!source.enabled || !account || source.microsoftRevision !== account.revision || source.user !== account.email) throw new Error('This Microsoft account changed or was disconnected. Reconnect and sync again.')
      if (account.expiresAt <= this.now() + 60_000) {
        const { response, json } = await this.token({ client_id: account.clientId, refresh_token: account.refreshToken, grant_type: 'refresh_token' })
        this.current(epoch)
        if (!response.ok) throw new Error('Microsoft could not refresh this account. Reconnect it or review your tenant policy.')
        const credentials = this.credentials(json, account)
        credentials.scopes = credentials.scopes.filter(scope => account!.scopes.includes(scope))
        account = { ...account, ...credentials }
        await this.commit(this.state.accounts.map(item => item.id === account!.id ? account! : item), epoch)
      }
      this.current(epoch)
      this.deps.assertSourceCurrent?.(source)
      return action(account, () => { this.current(epoch); this.deps.assertSourceCurrent?.(source) })
    })
  }
}
