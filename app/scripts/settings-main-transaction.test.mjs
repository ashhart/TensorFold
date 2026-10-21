import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
const studio = path.resolve(import.meta.dirname, '..');
const require = createRequire(path.join(studio, 'package.json'));
const {
  createJiti
} = require('jiti');
const tmp = process.argv[3] || (await fs.mkdtemp(path.join(os.tmpdir(), 'studio-settings-main-')));
const previous = process.env.VONTRA_HOME;
process.env.VONTRA_HOME = path.join(tmp, 'profile');
if (!process.argv[2]) {
  try {
    for (const mode of ['primary', 'models', 'concurrent', 'admission', 'metadata', 'apply', 'bot-pause', 'remote-boundary', 'profile-prepare', 'profile-commit']) {
      const child = spawnSync(process.execPath, [import.meta.filename, mode, tmp], {
        cwd: studio,
        encoding: 'utf8',
        timeout: 30_000
      });
      assert.equal(child.error, undefined, mode + ' consumer timed out');
      assert.equal(child.status, 0, mode + ': ' + child.stderr);
    }
    console.log('actual settings IPC disk faults, ordered snapshots, update admission, metadata conflicts and apply failures passed');
  } finally {
    if (previous === undefined) delete process.env.VONTRA_HOME;else process.env.VONTRA_HOME = previous;
    await fs.rm(tmp, {
      recursive: true,
      force: true
    });
  }
} else try {
  const actual = createJiti(path.join(studio, 'package.json'), {
    moduleCache: false,
    fsCache: false
  });
  const {
    defaultSettings,
    saveSettings,
    loadSettings
  } = await actual.import(path.join(studio, 'electron/store.ts'));
  const current = {
    ...defaultSettings(),
    activeModel: 'before',
    providers: [{
      id: 'synthetic-provider',
      name: 'Synthetic',
      baseUrl: 'https://model.example.test',
      apiKey: 'synthetic-before',
      models: ['model']
    }],
    engineProviderSeeded: true
  };
  await saveSettings(current);
  const {
    WorkspaceMutationGuard
  } = await actual.import(path.join(studio, 'electron/main/host/workspace-mutation-guard.ts'));
  const guard = new WorkspaceMutationGuard();
  const state = {
    normalize: async n => n,
    apply: async () => {},
    effects: 0,
    aborters: new Map(),
    vmHost: null,
    store: {
      settings: current,
      threads: []
    },
    handlers: new Map(),
    saveSettings,
    root: process.env.VONTRA_HOME
  };
  globalThis.__coreSettingsFixture = state;
  const fixture = path.join(tmp, 'fixture.mjs');
  await fs.writeFile(fixture, `import path from 'node:path';export const store=globalThis.__coreSettingsFixture.store;export const saveSettings=(s,options)=>globalThis.__coreSettingsFixture.saveSettings(s,options);export const vontraHomeDir=()=>globalThis.__coreSettingsFixture.root;export const vontraPath=(...p)=>path.join(vontraHomeDir(),...p);export const makeRemoteControlToken=()=>'synthetic-token';export const handle=(c,cb)=>globalThis.__coreSettingsFixture.handlers.set(c,cb);export const requireSecurityRole=()=>{};export const auditEvent=async()=>{};export const shell={};export const getBotVmHost=()=>globalThis.__coreSettingsFixture.vmHost;`);
  const noops = path.join(tmp, 'noops.mjs');
  const names = ['RemoteControl', 'Ssh', 'Workspace', 'Session', 'Spine', 'Mcp', 'Provider', 'GoogleWorkspace', 'Controller', 'Missions', 'Bot'];
  await fs.writeFile(noops, names.map(n => `export const register${n}Ipc=()=>{};`).join('\n') + `export const syncRemoteEventStreams=()=>{};export const sanitizeRendererThreadPatch=(x)=>x;export const openExternalSafely=async()=>{};export const modelRouterCooldownSnapshot=()=>({});export const securityControlSnapshot=()=>({});`);
  const alias = {
    '../../store/ui-preferences': path.join(studio, 'electron/store/ui-preferences.ts'),
    '../../store/model-helper-defaults': path.join(studio, 'electron/store/model-helper-defaults.ts'),
    '../../store': fixture,
    '../state': fixture,
    './audit': fixture,
    './ipc': fixture,
    './bot-vm-host': fixture,
    electron: fixture,
    './external': noops,
    './provider-runtime': noops,
    './security-controls': noops
  };
  for (const n of ['remote-control', 'ssh', 'workspace', 'session', 'spine', 'mcp', 'provider', 'google-workspace', 'controller', 'missions', 'bot']) alias['./ipc/' + n + '-ipc'] = noops;
  const jiti = createJiti(path.join(studio, 'package.json'), {
    alias,
    moduleCache: false,
    fsCache: false
  });
  const {
    registerCoreIpc
  } = await jiti.import(path.join(studio, 'electron/main/host/core-ipc.ts'));
  const noop = () => {};
  registerCoreIpc({
    aborters: state.aborters,
    workspaceMutationGuard: guard,
    registerConnectorIpc: noop,
    normalizeRemoteOrchestratorDestinations: (n, c) => state.normalize(n, c),
    ensureRemoteControlToken: noop,
    reloadPlugins: () => state.apply(),
    applyTheme: () => {
      state.effects++;
    },
    applyPowerSettings: noop,
    remoteServer: {
      syncFromSettings: async () => {}
    },
    startUpdateChecks: noop,
    startBotControlPolling: noop,
    syncWebhookServer: async () => {}
  });
  const mode = process.argv[2];
  const commit = next => state.handlers.get('settings:save')({}, next);
  const next = name => ({
    ...current,
    activeModel: name
  });
  if (mode === 'profile-prepare' || mode === 'profile-commit') {
    const firstRoot = state.root, secondRoot = path.join(tmp, 'other-profile');
    process.env.VONTRA_HOME = secondRoot;
    await saveSettings({ ...current, activeModel: 'other-profile-model' });
    process.env.VONTRA_HOME = firstRoot;
    const changeProfile = () => { state.root = secondRoot; process.env.VONTRA_HOME = secondRoot; };
    if (mode === 'profile-prepare') state.normalize = async n => { changeProfile(); return n; };
    else state.saveSettings = async (n, options) => { assert.equal(typeof options?.validateCommit, 'function', 'IPC passes its profile fence to the protected writer'); changeProfile(); return saveSettings(n, options); };
    await assert.rejects(() => commit(next('cross-profile-write')), /settings(?: profile)? changed/i);
    assert.equal(state.store.settings, current, 'stale profile save never replaces the live snapshot');
    assert.equal(state.effects, 0, 'stale profile save never applies runtime effects');
    assert.equal((await loadSettings()).activeModel, 'other-profile-model', 'other profile settings remain untouched');
    process.env.VONTRA_HOME = firstRoot; state.root = firstRoot;
    assert.equal((await loadSettings()).activeModel, 'before', 'original profile settings remain untouched');
  } else if (mode === 'remote-boundary') {
    const remote = { enabled: true, scope: 'selected', threadIds: ['allowed-chat'], roomIds: [], revision: 3, sessions: [{ ownerKey: 'synthetic-owner', authorizationRevision: '3', target: { kind: 'chat', id: 'allowed-chat' }, lastUpdateId: 12, updatedAt: 1, recentMessageTimes: [] }] };
    current.botControl.telegram.remote = remote;
    state.store.settings = current;
    const { redactSettings } = await actual.import(path.join(studio, 'electron/agent/security.ts'));
    const renderer = redactSettings(current);
    assert.deepEqual(renderer.botControl.telegram.remote.sessions, [], 'renderer receives no remote session state');
    assert.deepEqual(current.botControl.telegram.remote.sessions, remote.sessions, 'redaction does not clear host sessions');
    await commit({ ...renderer, activeModel: 'saved-other-setting', botControl: { ...renderer.botControl, telegram: { ...renderer.botControl.telegram, remote: { enabled: true, scope: 'all-eligible', threadIds: [], roomIds: [], revision: 999, sessions: [] } } } });
    assert.deepEqual(state.store.settings.botControl.telegram.remote, remote, 'generic settings cannot expand remote access or erase replay state');
    assert.deepEqual((await loadSettings()).botControl.telegram.remote, remote, 'owner-scoped remote policy remains durable');
    assert.equal(state.store.settings.activeModel, 'saved-other-setting');
  } else if (mode === 'bot-pause') {
    const preset = { id: 'bot_fixture', name: 'Fixture bot', role: 'Review', lane: 'review', providerId: 'synthetic-provider', model: 'model', executionTarget: 'vm', createdAt: 1, updatedAt: 1 };
    await commit({ ...current, agentPresets: [preset] });
    const controller = new AbortController();
    state.aborters.set('bot-chat', controller);
    const conversation = { id: 'bot-chat', bot: { presetId: preset.id, executionTarget: 'vm' }, timeline: [{ kind: 'user', text: 'Synthetic history remains.' }] };
    state.store.threads = [conversation];
    let stopped = 0;
    state.vmHost = { stop: async id => { assert.equal(id, preset.id); assert.equal(controller.signal.aborted, true, 'bot work is cancelled before the guest stops'); stopped++; } };
    await commit({ ...state.store.settings, agentPresets: [{ ...preset, paused: true }] });
    assert.equal(stopped, 1);
    assert.equal(state.store.settings.agentPresets[0].paused, true);
    assert.equal((await loadSettings()).agentPresets[0].paused, true, 'pause is durable');
    assert.equal(state.store.threads[0], conversation, 'pause retains conversation history');
    assert.equal(conversation.timeline.length, 1);
  } else if (mode === 'primary' || mode === 'models') {
    const blocked = path.join(state.root, mode === 'models' ? 'models.json' : 'settings.json');
    await fs.rename(blocked, blocked + '.before');
    await fs.mkdir(blocked);
    await assert.rejects(() => commit(next('after')), /EISDIR|ENOTDIR/);
    assert.equal(state.store.settings.activeModel, 'before', 'a rejected write cannot publish unpersisted live settings');
    assert.equal(state.effects, 0);
    if (mode === 'models') assert.equal(JSON.parse(await fs.readFile(path.join(state.root, 'settings.json'), 'utf8')).activeModel, 'before', 'a rejected derived-cache write cannot commit canonical settings');
    await fs.rmdir(blocked);
    await fs.rename(blocked + '.before', blocked);
    await commit(next('durable-retry'));
    assert.equal(state.store.settings.activeModel, 'durable-retry', 'failure releases the settings queue');
  } else if (mode === 'concurrent' || mode === 'admission') {
    let resume, entered;
    const gate = new Promise(r => resume = r),
      start = new Promise(r => entered = r);
    const order = [];
    state.normalize = async n => {
      order.push(n.activeModel);
      if (n.activeModel === 'alpha') {
        entered();
        await gate;
      }
      return n;
    };
    const a = commit({
      ...next('alpha'),
      providers: current.providers.map(p => ({
        ...p,
        apiKey: 'synthetic-after'
      }))
    });
    await start;
    assert.equal(state.store.settings.activeModel, 'before', 'pending save does not apply uncommitted settings');
    if (mode === 'concurrent') {
      const {
        redactSettings
      } = await actual.import(path.join(studio, 'electron/agent/security.ts'));
      const b = commit({
        ...redactSettings(current),
        activeModel: 'beta'
      });
      await Promise.resolve();
      assert.deepEqual(order, ['alpha'], 'second normalization waits for the first transaction');
      resume();
      await Promise.all([a, b]);
      assert.equal(state.store.settings.activeModel, 'beta');
      assert.equal(state.store.settings.providers[0].apiKey, 'synthetic-after', 'later masked snapshot preserves newly saved secret');
      assert.equal((await loadSettings()).activeModel, 'beta');
    } else {
      let acquired = false;
      const global = guard.acquireGlobal('App update installation').then(l => {
        acquired = true;
        return l;
      });
      await assert.rejects(() => commit(next('blocked')), /update installation/i);
      assert.equal(acquired, false, 'updater waits for an admitted settings save');
      resume();
      await a;
      const lease = await global;
      await assert.rejects(() => commit(next('still-blocked')), /update installation/i);
      lease.release();
      await commit(next('after-update'));
      assert.equal(state.store.settings.activeModel, 'after-update');
    }
  } else if (mode === 'metadata') {
    let resume, entered;
    const gate = new Promise(r => resume = r),
      start = new Promise(r => entered = r);
    state.saveSettings = async (n, options) => {
      entered();
      await gate;
      return saveSettings(n, options);
    };
    const pending = commit(next('after'));
    await start;
    state.store.settings = {
      ...current,
      approvalRules: [{
        id: 'new-authoritative-rule',
        scope: 'tool',
        key: 'test',
        decision: 'deny',
        duration: 'always',
        createdAt: 1,
        updatedAt: 1,
        workspace: 'synthetic',
        expiresAt: Date.now() + 10000,
        maxUses: 1,
        useCount: 0
      }]
    };
    await saveSettings(state.store.settings);
    resume();
    await assert.rejects(() => pending, /settings(?: profile)? changed/i);
    assert.equal(state.store.settings.activeModel, 'before');
    const persisted = await loadSettings();
    assert.equal(persisted.activeModel, 'before');
    assert.equal(persisted.approvalRules[0].id, 'new-authoritative-rule', 'concurrent metadata remains authoritative on disk');
  } else if (mode === 'apply') {
    state.apply = async () => {
      throw new Error('synthetic apply failure');
    };
    await assert.rejects(() => commit(next('after')), /settings were saved/i);
    assert.equal(state.store.settings.activeModel, 'after');
    assert.equal((await loadSettings()).activeModel, 'after', 'post-commit apply error states committed authority explicitly');
  }
  console.log('PASS ' + mode + ' settings transaction');
} finally {
  delete globalThis.__coreSettingsFixture;
  if (previous === undefined) delete process.env.VONTRA_HOME;else process.env.VONTRA_HOME = previous;
  await fs.rm(tmp, {
    recursive: true,
    force: true
  });
}
