// Wayland cannot present Vulkan surfaces, so disable only that feature, merging existing switches.

export interface CommandLineLike {
  hasSwitch(name: string): boolean
  getSwitchValue(name: string): string
  appendSwitch(name: string, value: string): void
}

export function isWaylandSession(platform: string, env: NodeJS.ProcessEnv, ozonePlatform: string): boolean {
  if (platform !== 'linux') return false
  if (ozonePlatform === 'wayland') return true
  if (ozonePlatform) return false
  return env.XDG_SESSION_TYPE === 'wayland' || !!env.WAYLAND_DISPLAY
}

/** Merge `Vulkan` into a comma-separated --disable-features value without dropping entries. */
export function withVulkanDisabled(existing: string | undefined): string {
  const entries = String(existing ?? '').split(',').map((entry) => entry.trim()).filter(Boolean)
  if (!entries.some((entry) => entry.toLowerCase() === 'vulkan')) entries.push('Vulkan')
  return entries.join(',')
}

export function configureLinuxGraphics(commandLine: CommandLineLike, platform = process.platform, env: NodeJS.ProcessEnv = process.env): void {
  if (!isWaylandSession(platform, env, commandLine.getSwitchValue('ozone-platform'))) return
  commandLine.appendSwitch('disable-features', withVulkanDisabled(commandLine.getSwitchValue('disable-features')))
}

/** Electron's app object, narrowed to the command line this module touches. */
export function applyLinuxGraphics(app: { commandLine: CommandLineLike }, platform = process.platform, env: NodeJS.ProcessEnv = process.env): void {
  configureLinuxGraphics(app.commandLine, platform, env)
}
