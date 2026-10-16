import type { PhoneActionInput, PhoneDeveloperConnectInput, PhonePairInput, PhoneSnapshot } from '../types'

// All display content is synthetic. Browser previews never discover or control
// a real device, request native permissions, or retain pairing credentials.
export function createMockPhoneApi() {
  const listeners = new Set<(snapshot: PhoneSnapshot) => void>()
  let counter = 0
  let page = 'home'
  let draft = ''
  let grantTimer: ReturnType<typeof setTimeout> | null = null
  const clearGrantTimer = () => { if (grantTimer) clearTimeout(grantTimer); grantTimer = null }
  let snapshot: PhoneSnapshot = {
    supported: true,
    adb: { available: true, path: '/demo/platform-tools/adb', version: 'Android Debug Bridge · simulated' },
    devices: [{ id: 'demo-phone-android', name: 'Demo Android', platform: 'android', transport: 'usb', state: 'ready', detail: 'Simulated phone for the browser preview.' }],
    session: null, observation: null, busy: false, error: null,
    mirroring: { available: false, detail: 'Native iPhone Mirroring is available only in the desktop app on a compatible Mac.' },
  }
  const read = () => structuredClone(snapshot)
  const publish = () => { for (const notify of listeners) notify(read()); return read() }
  const capture = () => {
    if (!snapshot.session) throw new Error('Connect a simulated phone first.')
    const safe = (text: string) => text.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[char]!))
    const title = page === 'home' ? 'Your day' : page === 'notes' ? 'Notes' : 'Settings'
    const body = page === 'notes' ? draft || 'A place for a new idea.' : page === 'settings' ? 'Phone controls are simulated.' : 'A quiet space to get things done.'
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="360" height="720" viewBox="0 0 360 720"><rect width="360" height="720" rx="28" fill="#10131a"/><text x="26" y="35" fill="#c4c9d3" font-family="system-ui" font-size="13">9:41</text><rect x="294" y="22" width="35" height="12" rx="4" stroke="#c4c9d3" fill="none"/><rect x="297" y="25" width="25" height="6" rx="2" fill="#c4c9d3"/><text x="26" y="122" fill="#eef0f5" font-family="system-ui" font-size="30" font-weight="600">${title}</text><text x="26" y="152" fill="#a6adbc" font-family="system-ui" font-size="13">${safe(body.slice(0, 40))}</text><rect x="24" y="195" width="312" height="132" rx="18" fill="#1b2230" stroke="#2e3748"/><text x="44" y="230" fill="#a7b3e9" font-family="system-ui" font-size="12">LOCAL PHONE PREVIEW</text><text x="44" y="264" fill="#edf0f6" font-family="system-ui" font-size="20">${page === 'notes' ? 'Draft note' : 'Notes'}</text><text x="44" y="290" fill="#a6adbc" font-family="system-ui" font-size="13">${page === 'notes' ? safe((draft || 'Nothing typed yet').slice(0, 34)) : 'Write and review a local note.'}</text><rect x="24" y="345" width="312" height="88" rx="18" fill="#191e28" stroke="#2e3748"/><text x="44" y="381" fill="#edf0f6" font-family="system-ui" font-size="18">Settings</text><text x="44" y="405" fill="#a6adbc" font-family="system-ui" font-size="13">Device preferences</text><text x="26" y="650" fill="#788195" font-family="system-ui" font-size="12">SIMULATED DEVICE · NO REAL PHONE ACCESS</text><rect x="135" y="695" width="90" height="4" rx="2" fill="#d4d8e3"/></svg>`
    snapshot = { ...snapshot, error: null, observation: { id: `demo-observation-${++counter}`, sessionId: snapshot.session.id, deviceId: snapshot.session.deviceId, capturedAt: Date.now(), width: 360, height: 720, image: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`, app: `demo.${page}`, warnings: ['Simulated screen; no real phone data is shown.'], nodes: [{ id: 'demo-notes', text: page === 'notes' ? draft || 'Draft note' : 'Notes', description: 'Open a synthetic note', role: page === 'notes' ? 'Text field' : 'Button', bounds: { x: 24, y: 195, width: 312, height: 132 }, editable: page === 'notes', secure: false }, { id: 'demo-settings', text: 'Settings', description: 'Open synthetic device preferences', role: 'Button', bounds: { x: 24, y: 345, width: 312, height: 88 }, editable: false, secure: false }] } }
    return publish()
  }
  return {
    phoneSnapshot: async () => read(),
    phoneRefresh: async () => publish(),
    phonePickAdb: async () => read(),
    phonePairWifi: async (input: PhonePairInput) => { if (!input.endpoint || !/^\d{6}$/.test(input.code)) throw new Error('Enter the pairing address and six-digit code shown on the phone.'); return publish() },
    phoneConnectWifi: async (endpoint: string) => {
      if (!endpoint.trim()) throw new Error('Enter the phone’s Wireless debugging address.')
      snapshot = { ...snapshot, devices: [...snapshot.devices.filter((device) => device.id !== 'demo-phone-wifi'), { id: 'demo-phone-wifi', name: 'Demo Android over Wi-Fi', platform: 'android', transport: 'wifi', state: 'ready', detail: 'Simulated paired connection.' }] }
      return publish()
    },
    phoneConnect: async ({ deviceId }: { deviceId: string }) => {
      if (snapshot.session) throw new Error('Disconnect the current phone before choosing another one.')
      const device = snapshot.devices.find((entry) => entry.id === deviceId)
      if (!device || device.state !== 'ready') throw new Error('Choose a ready phone.')
      snapshot = { ...snapshot, session: { id: `demo-session-${++counter}`, deviceId, name: device.name, platform: device.platform, transport: device.transport, threadId: null, actions: 0, startedAt: Date.now() }, observation: null, error: null }
      page = 'home'; draft = ''
      return publish()
    },
    phoneConnectDeveloper: async (input: PhoneDeveloperConnectInput) => {
      if (snapshot.session) throw new Error('Disconnect the current phone before choosing another one.')
      if (!input.endpoint.trim()) throw new Error('Enter a local WebDriverAgent endpoint.')
      const device = { id: 'demo-phone-ios', name: 'Demo iPhone', platform: 'ios' as const, transport: input.transport, state: 'ready' as const, detail: 'Simulated WebDriverAgent connection.' }
      snapshot = { ...snapshot, devices: [...snapshot.devices.filter((entry) => entry.id !== device.id), device], session: { id: `demo-session-${++counter}`, deviceId: device.id, name: device.name, platform: device.platform, transport: device.transport, threadId: null, actions: 0, startedAt: Date.now() }, observation: null, error: null }
      return publish()
    },
    phoneGrantChat: async (threadId: string) => {
      if (!snapshot.session || !threadId) throw new Error('Connect a phone and start a chat first.')
      if (snapshot.session.threadId && snapshot.session.threadId !== threadId) throw new Error('Stop the other chat’s phone control before allowing this chat.')
      clearGrantTimer()
      snapshot = { ...snapshot, session: { ...snapshot.session, threadId, grantExpiresAt: Date.now() + 15 * 60_000 }, observation: null }
      grantTimer = setTimeout(() => { grantTimer = null; snapshot = { ...snapshot, session: snapshot.session ? { ...snapshot.session, threadId: null, grantExpiresAt: null } : null, observation: null }; publish() }, 15 * 60_000)
      return publish()
    },
    phoneStop: async () => { clearGrantTimer(); snapshot = { ...snapshot, session: snapshot.session ? { ...snapshot.session, threadId: null, grantExpiresAt: null } : null, observation: null, busy: false, error: null }; return publish() },
    phoneDisconnect: async () => { clearGrantTimer(); snapshot = { ...snapshot, session: null, observation: null, busy: false, error: null }; draft = ''; return publish() },
    phoneObserve: async () => capture(),
    phoneAction: async (input: PhoneActionInput) => {
      if (!snapshot.session || !snapshot.observation || input.sessionId !== snapshot.session.id || input.observationId !== snapshot.observation.id || Date.now() - snapshot.observation.capturedAt > 30_000) throw new Error('Refresh the phone screen before acting.')
      if (input.action === 'home' || input.action === 'back') page = 'home'
      if (input.action === 'tap') page = input.y >= 345 ? 'settings' : 'notes'
      if (input.action === 'type') { page = 'notes'; draft = input.text.slice(0, 500) }
      if (input.action === 'launch') page = 'settings'
      snapshot = { ...snapshot, session: { ...snapshot.session, actions: snapshot.session.actions + 1 } }
      return capture()
    },
    phoneMirroringPermissions: async () => read(),
    onPhoneEvent: (callback: (snapshot: PhoneSnapshot) => void) => { listeners.add(callback); return () => { listeners.delete(callback) } },
  }
}
