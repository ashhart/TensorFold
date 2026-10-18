import assert from 'node:assert/strict'
import { isWhatsAppEventInCurrentScope } from '../../shared/whatsapp-event-scope'
import { makeWhatsAppBridgeSource, whatsappEventFromMessage } from './whatsapp-bridge'

const own = ['15551230000', 's.whatsapp.net'].join('@')
const contact = ['15551230001', 's.whatsapp.net'].join('@')
const stranger = ['15551230002', 's.whatsapp.net'].join('@')
const source = { ...makeWhatsAppBridgeSource('Current account'), phoneNumber: '+15551230000', allowedChatJids: [contact], allowSelfChat: true }
const event = (jid: string, selfChat = false) => whatsappEventFromMessage({ sourceId: source.id, jid, text: 'Synthetic fixture', sender: 'Fixture', fromMe: false, ts: 1, authenticatedAccountJid: own, selfChat })
assert.equal(isWhatsAppEventInCurrentScope(event(contact), [source]), true)
assert.equal(isWhatsAppEventInCurrentScope(event(stranger), [source]), false)
assert.equal(isWhatsAppEventInCurrentScope(event(contact), [{ ...source, allowedChatJids: [] }]), false, 'narrowed scope also hides previously cached selected-contact events')
assert.equal(isWhatsAppEventInCurrentScope(event(own, true), [source]), true)
assert.equal(isWhatsAppEventInCurrentScope(event(own, true), [{ ...source, allowSelfChat: false }]), false)
assert.equal(isWhatsAppEventInCurrentScope(event(own, true), [{ ...source, phoneNumber: '+15551239999' }]), false, 're-pairing an account never grants access to the previous phone’s self-chat')
assert.equal(isWhatsAppEventInCurrentScope(event(contact), [{ ...source, enabled: false }]), false)
assert.equal(isWhatsAppEventInCurrentScope(event(contact), []), false)
const unbound = event(contact)
delete unbound.payload!.authenticatedAccountJid
assert.equal(isWhatsAppEventInCurrentScope(unbound, [source]), false, 'legacy unbound cached events are not permission-bearing current context')
const fabricatedSelf = event(stranger, true)
assert.equal(isWhatsAppEventInCurrentScope(fabricatedSelf, [source]), false, 'self-chat flag alone never grants an unrelated phone identity')
const unrelatedLid = ['100000000099', 'lid'].join('@')
assert.equal(isWhatsAppEventInCurrentScope(event(unrelatedLid, true), [source]), false, 'a forged raw LID selfChat flag cannot impersonate the linked phone')
assert.equal(isWhatsAppEventInCurrentScope(event(unrelatedLid), [{ ...source, allowedChatJids: [unrelatedLid] }]), true, 'an exact selected non-self LID contact retains its ordinary grant')
const group = ['120363000001', 'g.us'].join('@')
assert.equal(isWhatsAppEventInCurrentScope(event(group), [{ ...source, allowedChatJids: [group] }]), true, 'cached selected group content retains account binding and exact current grant')
assert.equal(isWhatsAppEventInCurrentScope(event(group), [source]), false, 'unselected cached group content stays hidden')
assert.equal(isWhatsAppEventInCurrentScope(event(group, true), [{ ...source, allowedChatJids: [group] }]), false, 'a group can never turn into authenticated self-chat via metadata')
console.log('WhatsApp cached-event current account, exact scope and self-chat rebinding tests passed')
