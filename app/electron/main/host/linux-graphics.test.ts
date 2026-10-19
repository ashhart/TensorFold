// Run: node electron/main/host/linux-graphics.test.ts
import assert from 'node:assert/strict'
import { applyLinuxGraphics, configureLinuxGraphics, isWaylandSession, withVulkanDisabled } from './linux-graphics'

function fakeCommandLine(initial: Record<string, string> = {}) {
  const switches = new Map(Object.entries(initial))
  // Mirror Electron: getSwitchValue('') returns '' and hasSwitch reflects presence.
  return {
    switches,
    hasSwitch: (name: string) => switches.has(name),
    getSwitchValue: (name: string) => switches.get(name) ?? '',
    appendSwitch: (name: string, value: string) => { switches.set(name, value) },
  }
}

assert.equal(withVulkanDisabled(undefined), 'Vulkan', 'adds Vulkan when nothing was disabled')
assert.equal(withVulkanDisabled('Vulkan'), 'Vulkan', 'does not duplicate Vulkan')
assert.equal(withVulkanDisabled('vulkan'), 'vulkan', 'matches case-insensitively')
assert.equal(withVulkanDisabled('UseChromeOSDirectVideoDecoder'), 'UseChromeOSDirectVideoDecoder,Vulkan', 'keeps existing features')

assert.equal(isWaylandSession('linux', {}, 'wayland'), true, 'an explicit Wayland request counts')
assert.equal(isWaylandSession('linux', {}, 'x11'), false, 'an explicit X11 request is left alone')
assert.equal(isWaylandSession('linux', { XDG_SESSION_TYPE: 'wayland' }, ''), true)
assert.equal(isWaylandSession('linux', { WAYLAND_DISPLAY: 'wayland-0' }, ''), true)
assert.equal(isWaylandSession('linux', { XDG_SESSION_TYPE: 'x11' }, ''), false)
assert.equal(isWaylandSession('darwin', { XDG_SESSION_TYPE: 'wayland' }, ''), false, 'macOS is untouched')
assert.equal(isWaylandSession('linux', { XDG_SESSION_TYPE: 'wayland' }, 'x11'), false, 'a user forcing X11 keeps X11')

const explicit = fakeCommandLine({ 'ozone-platform': 'wayland' })
configureLinuxGraphics(explicit, 'linux', {})
assert.equal(explicit.switches.get('disable-features'), 'Vulkan', 'Wayland gets Vulkan disabled')

const session = fakeCommandLine()
configureLinuxGraphics(session, 'linux', { XDG_SESSION_TYPE: 'wayland' })
assert.equal(session.switches.get('disable-features'), 'Vulkan', 'a Wayland session gets the same treatment')

const merged = fakeCommandLine({ 'ozone-platform': 'wayland', 'disable-features': 'UseChromeOSDirectVideoDecoder' })
configureLinuxGraphics(merged, 'linux', {})
assert.equal(merged.switches.get('disable-features'), 'UseChromeOSDirectVideoDecoder,Vulkan', 'existing disable-features survive')

const x11 = fakeCommandLine({ 'ozone-platform': 'x11' })
configureLinuxGraphics(x11, 'linux', { XDG_SESSION_TYPE: 'wayland' })
assert.equal(x11.switches.has('disable-features'), false, 'X11 sessions are not changed')

const mac = fakeCommandLine()
configureLinuxGraphics(mac, 'darwin', { XDG_SESSION_TYPE: 'wayland' })
assert.equal(mac.switches.has('disable-features'), false, 'macOS is not changed')

applyLinuxGraphics({ commandLine: session }, 'linux', { XDG_SESSION_TYPE: 'wayland' })
assert.equal(session.switches.get('disable-features'), 'Vulkan', 'the app-facing helper applies the same rule')

console.log('linux graphics compatibility tests passed')
