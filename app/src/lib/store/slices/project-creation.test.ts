import assert from 'node:assert/strict'
const globals = globalThis as Record<string, unknown>
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}
globals.removeEventListener = () => {}
globals.window = globals
const { api } = await import('../../ipc')
const { useStore } = await import('../index')
const settings = await api.getSettings()
useStore.setState({ settings, projects: [], newProjectOpen: false, newProjectIntent: 0, view: { name: 'home' } })
const originalCreate = api.createProject
let release!: () => void, entered!: () => void
const pending = new Promise<void>(resolve => { release = resolve })
const started = new Promise<void>(resolve => { entered = resolve })
try {
  api.createProject = async (opts) => { entered(); await pending; return originalCreate(opts) }
  useStore.getState().openNewProject()
  const intent = useStore.getState().newProjectIntent
  const older = useStore.getState().newProject({ name: 'QA earlier save', mode: 'solo' }, intent)
  await started
  useStore.getState().closeNewProject()
  useStore.getState().openNewProject()
  release()
  const created = await older
  assert.ok(useStore.getState().projects.some(project => project.id === created.id), 'The already requested project still saves')
  assert.equal(useStore.getState().newProjectOpen, true, 'Earlier completion preserves a later reopened creation intent')
  assert.equal(useStore.getState().view.name, 'home', 'Earlier completion does not take navigation from the new intent')

  api.createProject = originalCreate
  const current = await useStore.getState().newProject({ name: 'QA current save', mode: 'solo' }, useStore.getState().newProjectIntent)
  assert.equal(useStore.getState().newProjectOpen, false)
  assert.deepEqual(useStore.getState().view, { name: 'project', id: current.id })
  const direct = await useStore.getState().newProject({ name: 'QA nonmodal save', mode: 'solo' })
  assert.deepEqual(useStore.getState().view, { name: 'project', id: direct.id }, 'Nonmodal callers keep their existing navigation behavior')

  useStore.getState().openNewProject()
  api.createProject = async () => { throw new Error('Synthetic creation failure') }
  await assert.rejects(() => useStore.getState().newProject({ name: 'QA rejected save', mode: 'solo' }, useStore.getState().newProjectIntent), /Synthetic creation failure/)
  assert.equal(useStore.getState().newProjectOpen, true, 'Failed creation remains reviewable and retryable')
} finally { release(); api.createProject = originalCreate }
console.log('Project creation keeps current modal ownership, preserves reopened drafts/navigation, supports nonmodal calls and remains retryable after rejection.')
