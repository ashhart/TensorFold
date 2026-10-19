import assert from 'node:assert/strict'
import { installDnsFixture } from '../../agent/test-fixtures/dns'
import { assertOutboundWebhookUrlResolved, isSafeExternalUrl, isSecretMask } from '../../agent/security'

// Regression guard for: "webhooks:save update branch mutated the live store entry
// BEFORE the https/SSRF validation throw". The fix builds a candidate (`next`),
// validates it, and only Object.assign's into the live `ex` once all checks pass.
//
// We re-create the exact update-branch sequence here (using the REAL security
// validators) and assert the live entry is left untouched when the URL is unsafe.

type Endpoint = {
  id: string
  direction: 'inbound' | 'outbound'
  url?: string
  path?: string
  secretRef: string
  eventKinds: string[]
  approvalRequired: boolean
  approvedAt: number | null
  updatedAt: number
}

// Mirrors the post-fix logic in webhook-ipc.ts handle('webhooks:save') update branch
// (now async: it also runs the DNS-resolved assertOutboundWebhookUrlResolved guard,
// same as the real handler, BEFORE the live entry is mutated).
async function applyUpdate(ex: Endpoint, w: Partial<Endpoint>, now: number): Promise<Endpoint> {
  const secretRef = isSecretMask(w.secretRef) ? ex.secretRef : (w.secretRef ?? ex.secretRef)
  const previousUrl = ex.url
  const next = Object.assign({}, ex, w, { secretRef, updatedAt: now }) as Endpoint
  if (next.direction === 'outbound') {
    next.eventKinds = Array.isArray(next.eventKinds)
      ? next.eventKinds.filter((k) => ['agent', 'room', 'ticket', 'automation', 'mission', 'connector', 'test'].includes(k))
      : []
    next.approvalRequired = next.approvalRequired ?? true
    if (w.url && w.url !== previousUrl) next.approvedAt = null
  } else {
    next.eventKinds = []
    next.approvalRequired = false
    next.approvedAt = null
  }
  if (next.direction === 'outbound' && next.url && !isSafeExternalUrl(next.url, { allowHttpLocalhost: true })) {
    throw new Error('Outbound webhook URL must use https, except localhost during development.')
  }
  if (next.direction === 'outbound' && next.url) {
    await assertOutboundWebhookUrlResolved(next.url)
  }
  Object.assign(ex, next)
  return ex
}

// --- SSRF guard (HIGH #6 / MEDIUM #20): bracketed private-IPv6 literal rejected ---------
// isSafeExternalUrl only enforces the scheme (https-or-localhost-http) — it has no concept
// of a private host at all, so a bracketed private-IPv6 literal URL like
// https://[fd12:3456::1]/hook passes it. assertOutboundWebhookUrlResolved is the guard that
// must reject it (used at both save time, here, and delivery time in webhooks.ts).
async function main() {
  // --- rejected edit must not mutate the live entry ---------------------------
  {
    const ex: Endpoint = {
      id: 'wh_1', direction: 'outbound', url: 'https://example.com/hook',
      secretRef: 'sec', eventKinds: ['agent'], approvalRequired: true,
      approvedAt: 1000, updatedAt: 1000,
    }
    const snapshot = { ...ex }
    let threw = false
    try {
      // unsafe http (non-localhost) URL edit — must be rejected
      await applyUpdate(ex, { url: 'http://evil.example.com/hook' }, 2000)
    } catch {
      threw = true
    }
    assert.equal(threw, true, 'unsafe URL edit should throw')
    assert.deepEqual(ex, snapshot, 'live entry must be UNCHANGED after a rejected edit (no partial mutation)')
    console.log('ok: rejected unsafe-URL edit leaves the live store entry untouched')
  }

  // --- rejected edit (private-IPv6 literal) must not mutate the live entry, either --------
  {
    const ex: Endpoint = {
      id: 'wh_1b', direction: 'outbound', url: 'https://example.com/hook',
      secretRef: 'sec', eventKinds: ['agent'], approvalRequired: true,
      approvedAt: 1000, updatedAt: 1000,
    }
    const snapshot = { ...ex }
    await assert.rejects(
      () => applyUpdate(ex, { url: 'https://[fd12:3456::1]/hook' }, 2000),
      /private or internal address/,
      'private-IPv6-literal URL edit should be rejected at save time',
    )
    assert.deepEqual(ex, snapshot, 'live entry must be UNCHANGED after a rejected private-IPv6 edit')
    console.log('ok: rejected private-IPv6 URL edit leaves the live store entry untouched')
  }

  // --- valid edit is fully applied -------------------------------------------
  {
    const ex: Endpoint = {
      id: 'wh_2', direction: 'outbound', url: 'https://example.com/old-hook',
      secretRef: 'sec', eventKinds: ['agent', 'bogus-kind'], approvalRequired: true,
      approvedAt: 5000, updatedAt: 1000,
    }
    await applyUpdate(ex, { url: 'https://example.com/new-hook' }, 9000)
    assert.equal(ex.url, 'https://example.com/new-hook', 'url updated')
    assert.equal(ex.updatedAt, 9000, 'updatedAt bumped')
    assert.equal(ex.approvedAt, null, 'changing the url re-arms approval (approvedAt cleared)')
    assert.deepEqual(ex.eventKinds, ['agent'], 'unknown event kinds filtered out')
    console.log('ok: valid edit is fully applied and normalized')
  }

  // --- masked secretRef preserves the existing secret -------------------------
  {
    const ex: Endpoint = {
      id: 'wh_3', direction: 'outbound', url: 'https://example.com/hook',
      secretRef: 'real-secret', eventKinds: ['agent'], approvalRequired: true,
      approvedAt: 1000, updatedAt: 1000,
    }
    // renderer sends back the redaction mask, not a real secret
    await applyUpdate(ex, { secretRef: '••••••••', enabled: true } as Partial<Endpoint> & { enabled: boolean }, 3000)
    assert.equal(ex.secretRef, 'real-secret', 'masked secretRef must keep the existing secret')
    console.log('ok: masked secretRef preserves the stored secret')
  }

  assert.equal(isSafeExternalUrl('https://[fd12:3456::1]/hook', { allowHttpLocalhost: true }), true,
    'isSafeExternalUrl alone does not catch private-IPv6 hosts — confirms the gap this guard closes')

  await assert.rejects(
    () => assertOutboundWebhookUrlResolved('https://[fd12:3456::1]/hook'),
    /private or internal address/,
    'bracketed IPv6 ULA (fd00::/8) literal is rejected',
  )
  await assert.rejects(
    () => assertOutboundWebhookUrlResolved('https://[fc00::1]/hook'),
    /private or internal address/,
    'bracketed IPv6 ULA (fc00::/8) literal is rejected',
  )
  await assert.rejects(
    () => assertOutboundWebhookUrlResolved('https://[fe80::1]/hook'),
    /private or internal address/,
    'bracketed IPv6 link-local literal is rejected',
  )
  console.log('ok: SSRF guard rejects bracketed private-IPv6 literal webhook URLs')

  // Existing IPv4/loopback behavior must still pass.
  await assert.rejects(
    () => assertOutboundWebhookUrlResolved('https://169.254.169.254/latest/meta-data/'),
    /private or internal address/,
    'IPv4 link-local/metadata literal still rejected',
  )
  await assert.rejects(
    () => assertOutboundWebhookUrlResolved('https://10.0.0.5/hook'),
    /private or internal address/,
    'IPv4 RFC1918 literal still rejected',
  )
  await assertOutboundWebhookUrlResolved('http://127.0.0.1:8765/hook') // loopback dev case still allowed
  await assertOutboundWebhookUrlResolved('http://localhost:8765/hook') // loopback dev case still allowed
  console.log('ok: existing IPv4/loopback webhook URL cases still pass')

  console.log('all webhook-ipc tests passed')
}

const restoreDns = installDnsFixture({ 'example.com': '8.8.8.8', localhost: '127.0.0.1' })
void main().finally(restoreDns)
