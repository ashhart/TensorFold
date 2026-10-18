import assert from 'node:assert/strict'
import clsx from 'clsx'
import { editQueuedFollowUp } from '../../lib/queued-followups'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'
import type { Steer } from './SteerQueueList'
const hooks = harness(), signals: (string | null)[] = [], guides: any[] = [], removals: any[] = []
let rows: Steer[] = [{ id: 'before', text: 'earlier', status: 'queued' }, { id: 'editing', text: 'original follow-up', status: 'queued' }, { id: 'after', text: 'later', status: 'queued' }]
let running = true
let edits = 0, reject = false, pending!: () => void
const { QueuedFollowUpRow } = load('src/components/composer/SteerQueueList.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, clsx, '../../lib/icons': { Icon: 'icon' } })
const onEdit = async (id: string, expectedText: string, nextText: string) => {
  edits++; await new Promise<void>(resolve => { pending = resolve })
  if (reject) return false
  const result = editQueuedFollowUp(rows, { id, expectedText, text: nextText }); if (result.ok) rows = result.queued; return result.ok
}
const render = () => { const tree = hooks.render(() => QueuedFollowUpRow({ q: rows[1], liveSteering: true, running, onGuide: (...args: any[]) => guides.push(args), onRemove: (id: string) => removals.push(id), onEdit, onEditingChange: (id: string | null) => signals.push(id) })); hooks.flush(); return tree }
const button = (label: string) => { const node = find(render(), row => row.type === 'button' && (row.props['aria-label'] === label || text(row) === label)); assert.ok(node, label); return node }
running = false; assert.equal(button('Send follow-up').props.title, 'Send this queued follow-up as the next turn'); assert.equal(guides.length, 0)
running = true
button('Edit queued follow-up').props.onClick()
assert.deepEqual(signals, ['editing'], 'The owner is notified synchronously before the editor opens so auto-drain can be suspended')
let area = find(render(), node => node.type === 'textarea'); area.props.onChange({ target: { value: 'cancelled edit' } })
button('Cancel').props.onClick(); assert.equal(rows[1].text, 'original follow-up'); assert.deepEqual(signals, ['editing', null])
button('Edit queued follow-up').props.onClick(); area = find(render(), node => node.type === 'textarea'); area.props.onChange({ target: { value: 'saved edit\n  indentation' } })
const save = button('Save changes'); save.props.onClick(); save.props.onClick(); assert.equal(edits, 1)
assert.equal(button('Cancel').props.disabled, true)
pending(); await settle(); assert.equal(rows[1].text, 'saved edit\n  indentation'); assert.deepEqual(rows.map(row => row.id), ['before', 'editing', 'after']); assert.equal(rows[1].status, 'queued')
assert.equal(guides.length, 0, 'Save never sends or guides a request'); assert.equal(removals.length, 0)
button('Edit queued follow-up').props.onClick(); area = find(render(), node => node.type === 'textarea'); area.props.onChange({ target: { value: 'typed against previous text' } })
rows = rows.map(row => row.id === 'editing' ? { ...row, text: 'changed elsewhere' } : row)
button('Save changes').props.onClick(); pending(); await settle()
assert.ok(text(render()).includes('Your edit remains here')); assert.equal(find(render(), node => node.type === 'textarea').props.value, 'typed against previous text'); assert.equal(rows[1].text, 'changed elsewhere')
button('Cancel').props.onClick()
button('Edit queued follow-up').props.onClick(); area = find(render(), node => node.type === 'textarea'); area.props.onChange({ target: { value: '   ' } }); assert.equal(button('Save changes').props.disabled, true)
area.props.onKeyDown({ key: 'Escape', nativeEvent: { isComposing: true }, preventDefault() { throw new Error('IME escape intercepted') } }); assert.ok(find(render(), node => node.type === 'textarea'))
area.props.onKeyDown({ key: 'Escape', nativeEvent: {}, preventDefault() {} }); assert.equal(find(render(), node => node.type === 'textarea'), undefined)
button('Edit queued follow-up').props.onClick(); rows = rows.map(row => row.id === 'editing' ? { ...row, status: 'guiding' } : row)
assert.equal(find(render(), node => node.type === 'textarea').props.disabled, true); assert.equal(button('Save changes').props.disabled, true)
assert.ok(text(render()).includes('already started'))
button('Cancel').props.onClick(); assert.equal(find(render(), node => node.type === 'button' && node.props['aria-label'] === 'Edit queued follow-up'), undefined)
rows = rows.map(row => row.id === 'editing' ? { ...row, status: 'guided' } : row)
assert.equal(find(render(), node => node.type === 'button' && node.props['aria-label'] === 'Edit queued follow-up'), undefined)
hooks.cleanup()
console.log('Actual queued follow-up editor has explicit save/cancel, synchronous auto-drain signals, CAS refusal with retained edits, preserved order/status, duplicate/blank/IME guards and no send side effects')
