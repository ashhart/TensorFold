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
    for (const mode of ['primary', 'models', 'concurrent', 'admission', 'metadata', 'apply']) {
      const child = spawnSync(process.execPath, [import.meta.filename, mode, tmp], {
        cwd: studio,
        encoding: 'utf8',
        timeout: 5000
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
    store: {
      settings: current
    },
    handlers: new Map(),
    saveSettings,
    root: process.env.VONTRA_HOME
  };
  globalThis.__coreSettingsFixture = state;
  const fixture = path.join(tmp, 'fixture.mjs');
  await fs.writeFile(fixture, `import path from 'node:path';export const store=globalThis.__coreSettingsFixture.store;export const saveSettings=(s)=>globalThis.__coreSettingsFixture.saveSettings(s);export const vontraPath=(...p)=>path.join(globalThis.__coreSettingsFixture.root,...p);export const makeRemoteControlToken=()=>'synthetic-token';export const handle=(c,cb)=>globalThis.__coreSettingsFixture.handlers.set(c,cb);export const requireSecurityRole=()=>{};export const auditEvent=async()=>{};export const shell={};`);
  const noops = path.join(tmp, 'noops.mjs');
  const names = ['RemoteControl', 'Ssh', 'Workspace', 'Session', 'Spine', 'Mcp', 'Provider', 'Controller', 'Missions', 'Bot'];
  await fs.writeFile(noops, names.map(n => `export const register${n}Ipc=()=>{};`).join('\n') + `export const syncRemoteEventStreams=()=>{};export const sanitizeRendererThreadPatch=(x)=>x;export const openExternalSafely=async()=>{};export const modelRouterCooldownSnapshot=()=>({});export const securityControlSnapshot=()=>({});`);
  const alias = {
    '../../store': fixture,
    '../state': fixture,
    './audit': fixture,
    './ipc': fixture,
    electron: fixture,
    './external': noops,
    './provider-runtime': noops,
    './security-controls': noops
  };
  for (const n of ['remote-control', 'ssh', 'workspace', 'session', 'spine', 'mcp', 'provider', 'controller', 'missions', 'bot']) alias['./ipc/' + n + '-ipc'] = noops;
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
  if (mode === 'primary' || mode === 'models') {
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
    state.saveSettings = async n => {
      entered();
      await gate;
      return saveSettings(n);
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
    await assert.rejects(() => pending, /settings changed/i);
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
