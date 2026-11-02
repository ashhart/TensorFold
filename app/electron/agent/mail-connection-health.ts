import type { MailConnectionAccount, MailConnectionHealth, MailConnectionTarget, MailConnectionsSnapshot, MailHealthState } from '../../shared/mail-connections'
import { createSerialTaskQueue } from './serial-task-queue'

export interface MailHealthRecord {
  key: string
  revision: string
  state: MailHealthState
  checkedAt: number
  lastSuccessAt: number | null
}
export interface MailHealthStore {
  read(): Promise<MailHealthRecord[]>
  write(records: MailHealthRecord[]): Promise<void>
}
interface HealthDeps {
  accounts(): Promise<{ googleConfigured: boolean; accounts: MailConnectionAccount[] }>
  probe(account: MailConnectionAccount): Promise<void>
  store: MailHealthStore
  now?: () => number
}
const MESSAGES: Record<MailHealthState, string> = {
  untested: 'Mailbox access has not been tested. Test connection performs a read only.',
  healthy: 'Mailbox read succeeded. No message was sent.',
  reconnect: 'Sign-in needs attention. Reconnect this account, then test again.',
  permission: 'Mailbox reading is not permitted. Review the account permissions and reconnect.',
  unreachable: 'The mail service could not be reached. Check your connection and try again.',
  'rate-limited': 'The mail service is rate limiting requests. Wait before testing again.',
  error: 'The mailbox check could not complete. Refresh the connection status and try again.',
  disconnected: 'This account is disconnected or disabled. Reconnect or enable it before testing.',
}
const key = (account: Pick<MailConnectionAccount, 'provider' | 'id'>) => `${account.provider}:${account.id}`
const identity = (account: MailConnectionAccount) => JSON.stringify([account.revision, account.email, account.authorization, account.read, account.send, account.inbox, account.sourceId])

function failure(reason: unknown): MailHealthState {
  const error = reason as { status?: number; message?: string }
  if (error?.status === 401 || /^(Microsoft could not refresh this account\.|Microsoft sign-in needs attention\.|Google sign-in|Google account is disconnected)/.test(error?.message ?? '')) return 'reconnect'
  if (error?.status === 403) return 'permission'
  if (error?.status === 429) return 'rate-limited'
  if (reason instanceof TypeError || /^(Could not reach Google\.|Google request was cancelled or timed out\.)/.test(error?.message ?? '')) return 'unreachable'
  return 'error'
}

export class MailConnectionHealthService {
  private readonly queue = createSerialTaskQueue()
  private readonly pending = new Map<string, Promise<MailConnectionHealth>>()
  constructor(private readonly deps: HealthDeps) { }

  private async records(): Promise<MailHealthRecord[]> {
    const records = await this.deps.store.read()
    if (!Array.isArray(records) || records.length > 200 || records.some(record => !record || typeof record.key !== 'string' || typeof record.revision !== 'string' || !Object.hasOwn(MESSAGES, record.state) || !Number.isFinite(record.checkedAt) || record.lastSuccessAt !== null && !Number.isFinite(record.lastSuccessAt))) throw new Error('Mail connection check history could not be read.')
    return records
  }

  private public(account: MailConnectionAccount, records: MailHealthRecord[]): MailConnectionHealth {
    const record = records.find(item => item.key === key(account) && item.revision === identity(account))
    const state = account.authorization === 'disconnected' ? 'disconnected' : account.authorization === 'reconnect' ? 'reconnect' : !account.read ? 'permission' : record?.state ?? 'untested'
    return { ...account, state, checkedAt: record?.checkedAt ?? null, lastSuccessAt: record?.lastSuccessAt ?? null, message: MESSAGES[state] }
  }

  async snapshot(): Promise<MailConnectionsSnapshot> {
    const [snapshot, records] = await Promise.all([this.deps.accounts(), this.records()])
    return { ...snapshot, accounts: snapshot.accounts.map(account => this.public(account, records)) }
  }

  test(target: MailConnectionTarget): Promise<MailConnectionHealth> {
    if (!target || !['gmail', 'microsoft'].includes(target.provider) || typeof target.id !== 'string' || !target.id || target.id.length > 160 || typeof target.revision !== 'string' || target.revision.length > 300) return Promise.reject(new Error('Choose a current mail account before testing.'))
    const requestKey = JSON.stringify([target.provider, target.id, target.revision])
    const existing = this.pending.get(requestKey)
    if (existing) return existing
    const job = this.check(target).finally(() => { if (this.pending.get(requestKey) === job) this.pending.delete(requestKey) })
    this.pending.set(requestKey, job)
    return job
  }

  private async current(target: MailConnectionTarget): Promise<MailConnectionAccount> {
    const account = (await this.deps.accounts()).accounts.find(item => key(item) === key(target))
    if (!account || account.revision !== target.revision) throw new Error('The mail account changed. Refresh its connection status before testing.')
    return account
  }

  private async check(target: MailConnectionTarget): Promise<MailConnectionHealth> {
    const account = await this.current(target)
    if (account.authorization !== 'connected' || !account.read) return this.public(account, await this.records())
    let state: MailHealthState = 'healthy'
    try { await this.deps.probe(account) } catch (reason) { state = failure(reason) }
    const live = await this.current(target)
    if (identity(account) !== identity(live)) throw new Error('The mail account changed during the check. Refresh its connection status.')
    return this.queue.run('records', async () => {
      const current = await this.current(target)
      if (identity(current) !== identity(account)) throw new Error('The mail account changed before the check could be saved.')
      const records = await this.records(), checkedAt = (this.deps.now ?? Date.now)()
      const prior = records.find(item => item.key === key(account) && item.revision === identity(account))
      const record: MailHealthRecord = { key: key(account), revision: identity(account), state, checkedAt, lastSuccessAt: state === 'healthy' ? checkedAt : prior?.lastSuccessAt ?? null }
      const next = [...records.filter(item => item.key !== record.key), record].slice(-200)
      await this.deps.store.write(next)
      const final = await this.current(target)
      if (identity(final) !== identity(account)) throw new Error('The mail account changed while saving the check. Refresh its connection status.')
      return this.public(final, next)
    })
  }
}
