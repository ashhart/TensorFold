// Typed, parseable routing identity for a connector conversation. Today threadKey is
// an opaque string ("slack:T1:C2:169...", "imessage:+447...") parsed ad-hoc all over
// the place. ThreadAddress turns it into a structured value the agent and the router
// can reason about: which platform, which account, which conversation, which message —
// plus burst-coalescing so a chatty thread is ONE triage signal, not N. Pure + tested.

export interface ThreadAddress {
  platform: string // imessage | slack | discord | whatsapp | email | telegram | ...
  account: string | null // the source/account (slack workspace, whatsapp number); null when single-account
  conversation: string // the channel / chat / thread id within the account
  message: string | null // a specific message/thread-ts when the key carries one
  raw: string
}

/**
 * Parse a threadKey ("<platform>:<a>:<b>:<c>") into a ThreadAddress. The number of
 * colon-parts after the platform decides the shape (matching how connectors build keys):
 *   1 part  -> conversation                       (imessage:<chat>, email:<root>)
 *   2 parts -> account, conversation              (discord:<src>:<chan>, whatsapp:<src>:<jid>)
 *   3 parts -> account, conversation, message     (slack:<src>:<chan>:<ts>)
 * Colons inside the last segment are preserved.
 */
export function parseThreadAddress(threadKey: string | null | undefined): ThreadAddress | null {
  const raw = (threadKey ?? '').trim()
  if (!raw) return null
  const i = raw.indexOf(':')
  if (i < 0) return { platform: raw, account: null, conversation: '', message: null, raw }
  const platform = raw.slice(0, i)
  const rest = raw.slice(i + 1)
  const parts = rest.split(':')
  if (parts.length === 1) return { platform, account: null, conversation: parts[0], message: null, raw }
  if (parts.length === 2) return { platform, account: parts[0], conversation: parts[1], message: null, raw }
  // 3+: account, conversation, then everything else is the message id (may contain ':').
  return { platform, account: parts[0], conversation: parts[1], message: parts.slice(2).join(':'), raw }
}

/** Rebuild a threadKey from a ThreadAddress (round-trips parseThreadAddress). */
export function formatThreadAddress(addr: Pick<ThreadAddress, 'platform' | 'account' | 'conversation' | 'message'>): string {
  return [addr.platform, addr.account, addr.conversation, addr.message].filter((p) => p != null && p !== '').join(':')
}

/** Same conversation = same platform + account + conversation (ignores the message id). */
export function sameConversation(a: ThreadAddress | null, b: ThreadAddress | null): boolean {
  if (!a || !b) return false
  return a.platform === b.platform && a.account === b.account && a.conversation === b.conversation
}

interface BurstEventLike { threadKey?: string | null; createdAt?: number }
export interface ThreadBurst<T> {
  address: ThreadAddress
  latest: T
  count: number
  // True when several messages landed within the coalesce window — one "chatty thread"
  // signal rather than N separate pings.
  burst: boolean
}

/**
 * Collapse events into one signal PER CONVERSATION (newest-first). `count` is how many
 * messages the conversation has; `burst` is true when >1 of them arrived within
 * `windowMs` of the latest — so a rapid back-and-forth is a single coalesced signal.
 */
export function coalesceThreadBursts<T extends BurstEventLike>(events: T[], windowMs = 60_000): ThreadBurst<T>[] {
  const groups = new Map<string, T[]>()
  for (const e of events) {
    const addr = parseThreadAddress(e.threadKey)
    const key = addr ? formatThreadAddress({ platform: addr.platform, account: addr.account, conversation: addr.conversation, message: null }) : (e.threadKey ?? '')
    const arr = groups.get(key)
    if (arr) arr.push(e)
    else groups.set(key, [e])
  }
  const out: ThreadBurst<T>[] = []
  for (const [key, arr] of groups) {
    const latest = arr.reduce((a, b) => ((b.createdAt ?? 0) > (a.createdAt ?? 0) ? b : a))
    const latestAt = latest.createdAt ?? 0
    const recent = arr.filter((e) => latestAt - (e.createdAt ?? 0) <= windowMs).length
    const address = parseThreadAddress(key) ?? { platform: key, account: null, conversation: '', message: null, raw: key }
    out.push({ address, latest, count: arr.length, burst: recent > 1 })
  }
  return out.sort((a, b) => (b.latest.createdAt ?? 0) - (a.latest.createdAt ?? 0))
}
