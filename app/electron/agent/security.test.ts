import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { auditPreview, buildSessionCsp, currentSecurityRoleFromEnv, securityRoleForLaunch, assertRemoteOrchestratorUrl, normalizeRemoteOrchestratorUrl, resolveRemoteOrchestratorUrl, assertLocalProviderDiscoveryUrl, assertLocalServiceUrlResolved, assertProviderEndpointResolved, assertConnectorBaseUrlResolved, isGloballyRoutableIp, isKnownMetadataIp, isPrivateIp, isPrivateHost, redactSettings, mergeSettingsSecrets, maskedSecretUrlError, SECRET_MASK } from './security'
import type { Provider, Settings } from './types'

async function rejects(raw: string, message: RegExp) {
  await assert.rejects(() => assertRemoteOrchestratorUrl(raw), message, raw)
}

// Minimal-but-complete Settings fixture (mirrors store.ts defaultSettings(), without the
// 'electron' import so it can run standalone via jiti). Only the connector blocks under
// test below carry non-empty values; everything else stays at a safe empty/default shape.
function baseSettings(): Settings {
  return {
    providers: [],
    activeProviderId: null,
    activeProviderAccountId: null,
    activeModel: null,
    agentPresets: [],
    theme: 'system',
    approvalMode: 'suggest',
    approvalRules: [],
    defaultCwd: null,
    temperature: 0.1,
    reasoningEffort: 'medium',
    autoCompact: true,
    contextLimit: 128_000,
    localContextLimit: 24_000,
    autoCheckUpdates: true,
    updateFeedUrl: '',
    power: { keepComputerAlive: false, lastStartedAt: null, lastError: null },
    email: { sources: [], maxMessages: 300, lastSyncAt: null },
    slack: { sources: [], maxMessages: 300, lastSyncAt: null },
    calendar: { sources: [], maxEvents: 500, lastSyncAt: null },
    documents: { sources: [], maxDocuments: 500, maxDocumentBytes: 1_000_000, lastSyncAt: null },
    browser: { sources: [], maxItems: 300, lastSyncAt: null },
    imessage: { enabled: false, allowedChatIds: '', maxMessages: 200, lastSyncAt: null, lastError: null },
    whatsapp: { sources: [], lastSyncAt: null },
    discord: { sources: [], lastSyncAt: null },
    mattermost: { sources: [], lastSyncAt: null },
    matrix: { sources: [], lastSyncAt: null },
    googlechat: { sources: [], lastSyncAt: null },
    botControl: {
      telegram: { enabled: false, botToken: '', allowedChatIds: '', allowedUserIds: '', polling: false, lastUpdateId: null, lastPollAt: null, lastError: null },
      whatsapp: { enabled: false, accessToken: '', phoneNumberId: '', verifyToken: '', appSecret: '', allowedChatIds: '', lastEventAt: null, lastError: null },
    },
    capabilities: { disabledCapabilityIds: [], health: {}, localScripts: [], packagedPlugins: [] },
    controller: { enabled: false, includeAccessibilityText: false, maxTextChars: 4_000, lastCapturedAt: null, lastError: null },
    voice: {
      enabled: true, ttsEnabled: true, sttEnabled: true, ttsProvider: 'browser', sttProvider: 'browser',
      voiceName: '', rate: 1, pitch: 1, volume: 1, autoSummarizeBeforeRead: true, summaryMaxChars: 900,
      autoReadRoomUpdates: false, apiBaseUrl: '', apiKey: '', ttsModel: '', sttModel: '',
      elevenLabsApiKey: '', elevenLabsVoiceId: '', elevenLabsModelId: '', elevenLabsStability: 0.4,
      elevenLabsStyle: 0.3, localTtsBaseUrl: '', localTtsVoice: '', localTtsEmotion: '',
    },
    remoteControl: {
      enabled: false, serverName: 'TensorFold Studio Orchestrator', bindHost: '127.0.0.1', port: 8765,
      authToken: '', allowLan: false, lastStartedAt: null, lastClientAt: null, lastError: null,
    },
    remoteClient: { enabled: false, serverUrl: '', authToken: '', lastConnectedAt: null, lastError: null },
    remoteOrchestrators: [],
    preset: 'tensorfold',
    light: { accent: '', background: '', foreground: '', uiFont: '', codeFont: '', translucent: false, contrast: 50 },
    dark: { accent: '', background: '', foreground: '', uiFont: '', codeFont: '', translucent: false, contrast: 50 },
    pointerCursors: true,
    sshHosts: [],
    trustedSshHostKeys: [],
  }
}

// One Mattermost/Matrix/Google Chat source carrying live secrets, used by the
// redactSettings/mergeSettingsSecrets coverage below.
function settingsWithConnectorSecrets(): Settings {
  const s = baseSettings()
  const now = Date.now()
  s.providers = [{ id: 'opencode-test', name: 'OpenCode', transport: 'opencode', baseUrl: '', apiKey: '', serverUrl: 'http://127.0.0.1:4096', serverPassword: 'opencode-server-password', models: ['test'] }]
  s.mattermost = {
    sources: [{
      id: 'mm-1', label: 'Mattermost', baseUrl: 'https://mm.example.com', token: 'mm-live-token-abc123',
      allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now,
    }],
    lastSyncAt: null,
  }
  s.matrix = {
    sources: [{
      id: 'mx-1', label: 'Matrix', baseUrl: 'https://matrix.org', token: 'mx-live-access-token-xyz789',
      allowedRooms: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now,
    }],
    lastSyncAt: null,
  }
  s.googlechat = {
    sources: [{
      id: 'gc-1', label: 'Google Chat', webhookUrl: 'https://chat.googleapis.com/v1/spaces/AAAA/messages?key=K&token=secret-webhook-token',
      accessToken: 'gc-live-access-token-456', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null,
      createdAt: now, updatedAt: now,
    }],
    lastSyncAt: null,
  }
  s.botControl.whatsapp = {
    ...s.botControl.whatsapp,
    appSecret: 'meta-live-app-secret-123',
  }
  return s
}

async function main() {
  assert.equal(currentSecurityRoleFromEnv({}), 'operator')
  assert.equal(currentSecurityRoleFromEnv({ VONTRA_SECURITY_ROLE: 'viewer' }), 'viewer')
  assert.equal(currentSecurityRoleFromEnv({ VONTRA_ROLE: 'owner' }), 'owner')
  assert.equal(currentSecurityRoleFromEnv({ VONTRA_SECURITY_ROLE: 'bogus' }), 'operator')

  const csp = buildSessionCsp()
  assert.match(csp, /script-src 'self'/)
  assert.match(csp, /object-src 'none'/)
  assert.doesNotMatch(csp, /connect-src[^;]*\shttps:/)
  assert.doesNotMatch(csp, /connect-src[^;]*\swss:/)
  // F1 (real-app QA): bracketed-IPv6 + ':*' port wildcard is rejected by Chromium
  // and only logs a console error — must not appear; localhost/127.0.0.1 cover local.
  assert.doesNotMatch(csp, /\[::1\]:\*/)
  assert.match(csp, /connect-src 'self'/)
  assert.doesNotMatch(csp, /connect-src[^;]*localhost:\*/)
  assert.match(csp, /trusted-types default vontra/)
  assert.match(csp, /require-trusted-types-for 'script'/)
  assert.match(csp, /frame-ancestors 'none'/)
  assert.match(csp, /frame-src[^;]*127\.0\.0\.1:5173/)
  assert.doesNotMatch(csp, /frame-src[^;]*localhost:\*/)
  assert.doesNotMatch(buildSessionCsp({ dev: true }), /require-trusted-types-for/)
  const devCsp = buildSessionCsp({ dev: true, connectSources: ['http://127.0.0.1:5173', 'ws://127.0.0.1:5173'] })
  assert.match(devCsp, /connect-src[^;]*http:\/\/127\.0\.0\.1:5173[^;]*ws:\/\/127\.0\.0\.1:5173/)
  const html = readFileSync('index.html', 'utf8')
  const metaCsp = html.match(/data-vontra-production-csp[^>]*content="([^"]+)"/)?.[1] ?? ''
  const directiveSources = (value: string, name: string) => value.split(';').map((part) => part.trim()).find((part) => part.startsWith(`${name} `))?.split(/\s+/).slice(1) ?? []
  assert.deepEqual(directiveSources(metaCsp, 'frame-src'), directiveSources(csp, 'frame-src'), 'production meta/header frame allowlists match exactly')
  assert.deepEqual(directiveSources(metaCsp, 'connect-src'), ["'self'"], 'production meta fallback allows only same-origin renderer fetches')
  assert.match(html, /<meta name="referrer" content="no-referrer"/)

  assert.equal(await assertRemoteOrchestratorUrl('http://127.0.0.1:8765/'), 'http://127.0.0.1:8765')
  assert.equal(await assertRemoteOrchestratorUrl('http://localhost:8765/base/'), 'http://localhost:8765/base')
  assert.equal(await assertRemoteOrchestratorUrl('https://10.0.0.5:8765/orchestrator/'), 'https://10.0.0.5:8765/orchestrator')
  assert.equal(await assertRemoteOrchestratorUrl('https://100.64.0.1:8765/'), 'https://100.64.0.1:8765', 'Tailscale CGNAT literals are supported over pinned HTTPS')
  assert.equal(normalizeRemoteOrchestratorUrl('https://studio.tailnet.example:8765/base/'), 'https://studio.tailnet.example:8765/base', 'lexical normalization does not require DNS')
  const magicDns = await resolveRemoteOrchestratorUrl('https://studio.tailnet.example:8765/', async () => [{ address: '100.72.1.9', family: 4 }])
  assert.equal(magicDns.baseUrl, 'https://studio.tailnet.example:8765', 'MagicDNS/private names are supported over strict HTTPS')

  await rejects('http://192.168.1.50:8765', /must use https/)
  await rejects('http://example.com:8765', /must use https/)
  await rejects('file:///tmp/vontra.sock', /must use http or https/)
  await rejects('https://token@example.com', /must not include credentials/)
  await rejects('https://127.0.0.1:8765/?token=abc', /must not include query/)
  await rejects('https://169.254.169.254/latest', /link-local or metadata/)

  // Local-provider discovery must only ever send the (renderer-supplied) apiKey to a
  // loopback/LAN box — a public host turns the handler into a credential-exfil sink.
  // Empty/loopback/private literals are allowed without DNS; public IP literals + bad
  // shapes are rejected before any fetch.
  await assertLocalProviderDiscoveryUrl('') // no baseUrl => built-in localhost default; OK
  await assertLocalProviderDiscoveryUrl('http://127.0.0.1:11434')
  await assertLocalProviderDiscoveryUrl('http://localhost:1234')
  await assertLocalProviderDiscoveryUrl('http://192.168.1.50:1234/v1') // LAN oMLX
  await assertLocalProviderDiscoveryUrl('http://10.0.0.9:8080')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://1.2.3.4:1234'), /not a private|refusing/i, 'public IP literal rejected')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://8.8.8.8/v1'), /not a private|refusing/i, 'public IP literal rejected')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://169.254.169.254/latest'), /link-local|metadata/i, 'metadata/link-local is never a valid local provider')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('ftp://127.0.0.1'), /http or https/i, 'bad protocol rejected')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://user:pass@127.0.0.1'), /credentials/i, 'embedded credentials rejected')

  // IPv4-mapped IPv6: Node's URL parser compresses [::ffff:169.254.169.254] to the hex form
  // [::ffff:a9fe:a9fe]. The guards must classify these by their embedded IPv4 or Bearer
  // credentials/tokens leak to internal/metadata hosts (gap-audit CRITICAL).
  assert.equal(isPrivateIp('::ffff:a9fe:a9fe'), true, 'mapped metadata 169.254.169.254')
  assert.equal(isPrivateIp('::ffff:c0a8:101'), true, 'mapped 192.168.1.1')
  assert.equal(isPrivateIp('::ffff:a00:1'), true, 'mapped 10.0.0.1')
  assert.equal(isPrivateIp('::ffff:192.168.1.1'), true, 'mapped dotted form still classified')
  assert.equal(isPrivateIp('::ffff:808:808'), false, 'mapped 8.8.8.8 is public')
  assert.equal(isKnownMetadataIp('100.100.100.200'), true, 'Alibaba metadata is always forbidden')
  assert.equal(isKnownMetadataIp('fd00:ec2:0:0:0:0:0:254'), true, 'AWS IPv6 metadata equivalent spelling is always forbidden')
  assert.equal(isGloballyRoutableIp('93.184.216.34'), true)
  assert.equal(isGloballyRoutableIp('198.18.0.1'), false, 'benchmark ranges are not public egress')
  assert.equal(isGloballyRoutableIp('64:ff9b::a9fe:a9fe'), false, 'well-known NAT64 is not treated as global public egress')
  assert.equal(isPrivateHost('[::ffff:a9fe:a9fe]'), true, 'host: mapped metadata is private')
  assert.equal(isPrivateHost('[::ffff:c0a8:101]'), true, 'host: mapped LAN is private')
  assert.equal(isPrivateHost('[::ffff:808:808]'), false, 'host: mapped public stays public')
  await rejects('https://[::ffff:169.254.169.254]/x', /link-local or metadata/) // orchestrator token SSRF

  // isPrivateHost must also recognize literal IPv6 ULA (fc00::/7) and link-local
  // (fe80::/10) hosts directly — not just via the IPv4-mapped form. Webhook SSRF guard
  // (HIGH #6) relies on this to reject e.g. https://[fd12:3456::1]/hook without DNS.
  assert.equal(isPrivateHost('[fd12:3456::1]'), true, 'host: IPv6 ULA (fd00::/8) is private')
  assert.equal(isPrivateHost('[fc00::1]'), true, 'host: IPv6 ULA (fc00::/8) is private')
  assert.equal(isPrivateHost('[fe80::1]'), true, 'host: IPv6 link-local (fe80::/10) is private')
  assert.equal(isPrivateHost('[2001:db8::1]'), false, 'host: public-looking IPv6 stays non-private')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://[::ffff:8.8.8.8]/v1'), /not a private|refusing/i, 'mapped public rejected for discovery')
  await assertLocalProviderDiscoveryUrl('http://[::ffff:192.168.1.5]:1234') // mapped LAN allowed

  const disguisedPublicLocal = {
    id: 'unsafe-local', name: 'Unsafe local', kind: 'local', baseUrl: 'http://8.8.8.8/v1', apiKey: 'secret', models: [],
  } as Provider
  await assert.rejects(() => assertProviderEndpointResolved(disguisedPublicLocal), /public address|mark public endpoints as cloud/i,
    'kind:local cannot downgrade an arbitrary public HTTP endpoint into the local trust policy')

  await assert.rejects(() => assertConnectorBaseUrlResolved('http://example.com', 'Connector'), /must use HTTPS/i,
    'public connector HTTP is rejected without a private-network opt-in')
  await assert.rejects(() => assertConnectorBaseUrlResolved('http://8.8.8.8', 'Connector', { allowPrivateNetwork: true }), /public address|must use HTTPS/i,
    'private-network opt-in cannot authorize plaintext HTTP to a public destination')
  await assert.rejects(() => assertConnectorBaseUrlResolved('http://100.100.100.200/latest', 'Connector', { allowPrivateNetwork: true }), /metadata|private|link-local/i,
    'private-network opt-in never permits Alibaba metadata')
  await assert.rejects(() => assertConnectorBaseUrlResolved('http://[fd00:ec2::254]/latest', 'Connector', { allowPrivateNetwork: true }), /metadata|private|link-local/i,
    'private-network opt-in never permits AWS IPv6 metadata')
  await assertLocalServiceUrlResolved('http://127.0.0.1:4096', 'OpenCode server')
  await assert.rejects(() => assertLocalServiceUrlResolved('http://8.8.8.8:4096', 'OpenCode server'), /HTTP only on loopback or a private network/i)
  await assert.rejects(() => assertLocalServiceUrlResolved('http://169.254.169.254', 'OpenCode server'), /link-local|metadata/i)

  // The local desktop GUI is the machine owner (may save settings/policies);
  // headless/remote servers keep the safer 'operator' default; explicit env wins.
  assert.equal(securityRoleForLaunch({}, { headless: false }), 'owner')
  assert.equal(securityRoleForLaunch({}, { headless: true }), 'operator')
  assert.equal(securityRoleForLaunch({ VONTRA_SECURITY_ROLE: 'viewer' }, { headless: false }), 'viewer')
  assert.equal(securityRoleForLaunch({ VONTRA_ROLE: 'operator' }, { headless: false }), 'operator')
  assert.equal(securityRoleForLaunch({ VONTRA_SECURITY_ROLE: 'bogus' }, { headless: false }), 'owner')

  const preview = auditPreview('Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz123456 apiKey=sk-live-secret123 password=hunter2hunter2 postgres://admin:s3cretp4ss@db.internal/app')
  assert.doesNotMatch(preview, /ghp_abcdefghijklmnopqrstuvwxyz123456/)
  assert.doesNotMatch(preview, /sk-live-secret123/)
  assert.doesNotMatch(preview, /hunter2hunter2/)
  assert.doesNotMatch(preview, /s3cretp4ss/)
  assert.match(preview, /••••••••/, 'audit previews redact obvious secrets before writing plaintext audit logs')
  const opaqueAuditSecret = 'q7Kp2_Zm9Ax4Vn8Ld3Rt6Yw1Bc5Hs0JuEeNf'
  assert.doesNotMatch(auditPreview(`probe=${opaqueAuditSecret}`), new RegExp(opaqueAuditSecret), 'auditPreview masks keyword-less opaque credentials by default')
  assert.match(auditPreview('commit a3f5c8d714e9b2c06a1d8f9433c7a012d439ef12'), /a3f5c8d714e9b2c06a1d8f9433c7a012d439ef12/, 'auditPreview preserves Git hashes')

  // [HIGH #4] redactSettings() must mask Mattermost/Matrix/Google Chat live secrets the
  // same way it already masks Slack/Discord tokens — otherwise they reach the renderer
  // (readable via DevTools) in plaintext on every settings:get.
  const live = settingsWithConnectorSecrets()
  const redacted = redactSettings(live)
  assert.equal(redacted.providers[0].serverPassword, SECRET_MASK, 'provider serverPassword masked')
  assert.equal(redacted.mattermost.sources[0].token, SECRET_MASK, 'mattermost token masked')
  assert.equal(redacted.matrix.sources[0].token, SECRET_MASK, 'matrix token masked')
  assert.equal(redacted.googlechat.sources[0].webhookUrl, SECRET_MASK, 'googlechat webhookUrl masked')
  assert.equal(redacted.googlechat.sources[0].accessToken, SECRET_MASK, 'googlechat accessToken masked')
  assert.equal(redacted.botControl.whatsapp.appSecret, SECRET_MASK, 'WhatsApp Meta App Secret masked')
  // Sanity: the live secrets really were present pre-redaction (fixture didn't accidentally
  // start empty, which would make the masking assertions above vacuously true).
  assert.notEqual(live.mattermost.sources[0].token, SECRET_MASK)
  assert.notEqual(live.matrix.sources[0].token, SECRET_MASK)
  assert.notEqual(live.googlechat.sources[0].webhookUrl, SECRET_MASK)
  assert.notEqual(live.googlechat.sources[0].accessToken, SECRET_MASK)

  // mergeSettingsSecrets() is the restore-on-save counterpart: when the renderer sends
  // back the mask UNCHANGED, the real persisted secret must be restored (not literally
  // saved as the mask string); when the renderer sends a genuinely NEW value, that new
  // value must be kept.
  const resavedUnchanged = redactSettings(live) // renderer round-trip: edits nothing, mask comes back as-is
  const restored = mergeSettingsSecrets(resavedUnchanged, live)
  assert.equal(restored.providers[0].serverPassword, live.providers[0].serverPassword, 'provider serverPassword restored from current')
  assert.equal(restored.mattermost.sources[0].token, live.mattermost.sources[0].token, 'mattermost token restored from current')
  assert.equal(restored.matrix.sources[0].token, live.matrix.sources[0].token, 'matrix token restored from current')
  assert.equal(restored.googlechat.sources[0].webhookUrl, live.googlechat.sources[0].webhookUrl, 'googlechat webhookUrl restored from current')
  assert.equal(restored.googlechat.sources[0].accessToken, live.googlechat.sources[0].accessToken, 'googlechat accessToken restored from current')
  assert.equal(restored.botControl.whatsapp.appSecret, live.botControl.whatsapp.appSecret, 'WhatsApp Meta App Secret restored from current')

  const edited = redactSettings(live)
  edited.providers[0].serverPassword = 'new-opencode-server-password'
  edited.mattermost.sources[0].token = 'mm-brand-new-token'
  edited.matrix.sources[0].token = 'mx-brand-new-token'
  edited.googlechat.sources[0].webhookUrl = 'https://chat.googleapis.com/v1/spaces/BBBB/messages?key=K&token=brand-new-webhook'
  edited.googlechat.sources[0].accessToken = 'gc-brand-new-token'
  edited.botControl.whatsapp.appSecret = 'meta-brand-new-secret'
  const mergedWithEdits = mergeSettingsSecrets(edited, live)
  assert.equal(mergedWithEdits.providers[0].serverPassword, 'new-opencode-server-password', 'genuinely new provider serverPassword kept')
  assert.equal(mergedWithEdits.mattermost.sources[0].token, 'mm-brand-new-token', 'genuinely new mattermost token kept')
  assert.equal(mergedWithEdits.matrix.sources[0].token, 'mx-brand-new-token', 'genuinely new matrix token kept')
  assert.equal(mergedWithEdits.googlechat.sources[0].webhookUrl, 'https://chat.googleapis.com/v1/spaces/BBBB/messages?key=K&token=brand-new-webhook', 'genuinely new googlechat webhookUrl kept')
  assert.equal(mergedWithEdits.googlechat.sources[0].accessToken, 'gc-brand-new-token', 'genuinely new googlechat accessToken kept')
  assert.equal(mergedWithEdits.botControl.whatsapp.appSecret, 'meta-brand-new-secret', 'genuinely new WhatsApp Meta App Secret kept')

  // RED-TEAM REGRESSIONS — compromised-renderer rules.
  // 1) A renderer must never grant packaged-plugin trust: plugins load via require() in
  //    the main process, so trust only carries over from persisted settings when the
  //    plugin id AND manifest path are unchanged.
  {
    const trusted = mergeSettingsSecrets({
      ...redactSettings(live),
      capabilities: { ...live.capabilities, packagedPlugins: [
        { id: 'evil', name: 'evil', enabled: true, trusted: true, manifestPath: '/Users/example/.cache/evil/vontra-plugin.json' },
        { id: 'good', name: 'good', enabled: true, trusted: true, manifestPath: '/opt/plugins/good/vontra-plugin.json' },
      ] } as unknown as typeof live,
    } as unknown as typeof live, {
      ...live,
      capabilities: { ...live.capabilities, packagedPlugins: [
        { id: 'good', name: 'good', enabled: true, trusted: true, manifestPath: '/opt/plugins/good/vontra-plugin.json' },
      ] } as unknown as typeof live,
    } as unknown as typeof live)
    assert.equal(trusted.capabilities?.packagedPlugins?.[0].trusted, false, 'a NEW plugin can never arrive trusted from the renderer')
    assert.equal(trusted.capabilities?.packagedPlugins?.[1].trusted, true, 'an unchanged id+manifestPath keeps its persisted trust')
    const moved = mergeSettingsSecrets({
      ...redactSettings(live),
      capabilities: { ...live.capabilities, packagedPlugins: [
        { id: 'good', name: 'good', enabled: true, trusted: true, manifestPath: '/Users/example/.cache/evil/good.json' },
      ] } as unknown as typeof live,
    } as unknown as typeof live, {
      ...live,
      capabilities: { ...live.capabilities, packagedPlugins: [
        { id: 'good', name: 'good', enabled: true, trusted: true, manifestPath: '/opt/plugins/good/vontra-plugin.json' },
      ] } as unknown as typeof live,
    } as unknown as typeof live)
    assert.equal(moved.capabilities?.packagedPlugins?.[0].trusted, false, 'a CHANGED manifest path resets trust')
  }

  // 2) A masked API key paired with a changed baseUrl is cleared, not restored: the
  //    stored key must never be silently re-pointed at a renderer-chosen host.
  {
    const base = redactSettings(live)
    const swapped = mergeSettingsSecrets({ ...base, providers: base.providers.map((p, i) => i === 0 ? { ...p, baseUrl: 'https://evil.example/v1' } : p) } as typeof live, live)
    assert.equal(swapped.providers[0].apiKey, '', 'masked key + changed baseUrl clears the key')
    const unchanged = mergeSettingsSecrets(base, live)
    assert.equal(unchanged.providers[0].apiKey, live.providers[0].apiKey, 'masked key with the SAME baseUrl still restores')
  }

  // 3) maskedSecretUrlError drives the provider:models / remote:testClient refusals.
  assert.equal(maskedSecretUrlError(true, 'https://evil.example', 'https://api.real.example'), 'A saved secret can only be used with its saved URL. Re-enter the secret (API key or token) to use a different URL.')
  assert.equal(maskedSecretUrlError(true, 'https://api.real.example', 'https://api.real.example'), null)
  assert.equal(maskedSecretUrlError(false, 'https://evil.example', 'https://api.real.example'), null, 'a freshly typed secret may target any URL')
  assert.equal(maskedSecretUrlError(true, '', 'https://api.real.example'), null)
  assert.equal(maskedSecretUrlError(true, 'https://api.real.example/', 'https://api.real.example'), null, 'trailing slashes do not count as a change')

  // RED-TEAM REGRESSIONS — redaction gaps found by the adversarial audit.
  assert.equal(auditPreview('Authorization: Basic YWRtaW46aHVudGVyMg=='), 'Authorization: Basic ••••••••')
  assert.equal(auditPreview('Cookie: session=abcdefgh123456; other=zzzz9'), 'Cookie: ••••••••')
  assert.equal(auditPreview('https://storage.example/file?Expires=123&Signature=abcdef0123456789abcdef0123456789'), `https://storage.example/file?Expires=123&Signature=${SECRET_MASK}`)
  assert.equal(auditPreview('https://s3.example/file?X-Amz-Signature=abcdef0123456789abcdef0123456789&X-Amz-Expires=60'), `https://s3.example/file?X-Amz-Signature=${SECRET_MASK}&X-Amz-Expires=60`)
  assert.equal(auditPreview('sha256 checksum abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123 stays readable'), auditPreview('sha256 checksum abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123abcd0123 stays readable'), 'checksum-style hex is still not masked wholesale')

  console.log('security tests passed')
}

void main()
