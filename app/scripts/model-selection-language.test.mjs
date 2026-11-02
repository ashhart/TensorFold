import fs from 'node:fs'
import path from 'node:path'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const app = process.cwd(), req = createRequire(import.meta.url)
const React = req('react'), { renderToStaticMarkup } = req('react-dom/server'), ts = req('typescript')
// Real React exercises the installed-model callback; synthetic native APIs never load a model.
function load(file, imports) {
 const module = { exports: {} };
 const output = ts.transpileModule(fs.readFileSync(path.join(app, file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
 new Function('require', 'module', 'exports', output)(id => { assert.ok(id in imports, 'Reviewed import '+id); return imports[id] }, module, module.exports);
 return module.exports;
}
const state = { settings: { uiPreferences: { locale: 'en' }, desktopOnboardingComplete: true }, threads: [], status: {}, view: { name: 'models' }, newChat: async () => {}, setNotice() {} };
const useStore = Object.assign(selector => selector(state), { getState: () => state });
const authoredText = (_locale, source) => source;
const { useAuthoredText } = load('src/lib/preferences/useAuthoredText.ts', { react: React, '../store': { useStore }, './language-packs': { authoredText } });
let click, loads = 0;
const empty = () => null;
const row = props => { click = props.onLoad; return null };
const engine = { ENGINE_PROVIDER_ID: 'tensorfold', useEngine: () => ({ snapshot: { status: { state: 'idle', available: true }, installed: [{ repo: 'synthetic/model', title: 'Synthetic', fitsMemory: true }], suggestions: [], pulls: [] } }), loadEngineModel: async () => { loads++ }, refreshEngine: async () => {} };
const { Models } = load('src/components/Models.tsx', {
 '../../shared/hub-failure': load('shared/hub-failure.ts', {}),
 react: React, 'react/jsx-runtime': req('react/jsx-runtime'), clsx: req('clsx'),
 './OriginalDiagnostic': { OriginalDiagnostic: empty }, '../lib/preferences/language-packs': { authoredText }, '../lib/preferences/useAuthoredText': { useAuthoredText }, '../lib/store': { useStore }, '../lib/ipc': { api: {} }, '../lib/engine': engine,
 './clusters/ClusterWorkspace': { ClusterWorkspace: empty },
 './models/DownloadActivity': { DownloadActivity: empty },
 '../lib/model-chat': { openModelChat: async () => {} },
 '../lib/favorite-models': { useFavoriteModels: () => ({ ready: true, favorites: [], isFavorite: () => false }), orderFavoritesFirst: items => items }, './ui': { Button: empty, Input: empty }, '../lib/icons': { Icon: empty }, './models/ModelRows': { HubRow: empty, InstalledRow: row }, './models/EngineRuntimePanel': { EngineRuntimePanel: empty }, './models/EngineApiPanel': { EngineApiPanel: empty }, '../lib/learning': { useLearning: () => ({ snapshot: null }), learningBusy: () => false }, './models/LearningPanel': { LearningPanel: empty }, './models/WorkersPanel': { WorkersPanel: empty }, './models/workers': { isLocalWorker: () => false }, '../lib/model-recommendation': { recommendEngineModel: () => ({ selected: null }) }, './models/ModelRecommendationPanel': { ModelRecommendationPanel: empty }, './onboarding-model': { shouldShowOnboarding: () => false }, './models/ModelDetailsDrawer': { ModelDetailsDrawer: empty }, './models/ModelComparisonPanel': { ModelComparisonPanel: empty }, './models/ModelsDrawer': { ModelsDrawer: empty }, './models/DownloadCentre': { DownloadCentre: empty }, './models/download-display': { isDownloadPending: () => false, downloadStateLabel: () => '' }, './settings/ModelFoldersSettings': { ModelFoldersSettings: empty }, '../lib/turn-stats-line': { repoDisplayName: x => x }, './models/benchmark/ModelBenchmarkDrawer': { ModelBenchmarkDrawer: empty },
});
(async () => {
 renderToStaticMarkup(React.createElement(Models));
 assert.equal(typeof click, 'function');
 const rejection = await new Promise(resolve => {
  const rejected = error => { clearTimeout(timer); resolve(error) };
  const timer = setTimeout(() => { process.removeListener('unhandledRejection', rejected); resolve(null) }, 30);
  process.once('unhandledRejection', rejected); click();
 });
 assert.equal(rejection, null, 'Selecting an installed model must not invoke a React hook from its event handler');
 assert.equal(loads, 1, 'The existing model loader should run once');
 console.log('Actual Models installed-row callback invokes the loader without calling hooks outside render.');
})().catch(error => { console.error(error); process.exitCode = 1 });
