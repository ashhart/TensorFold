export interface PackageInstallCommand {
  manager: 'npm'
  operation: 'install' | 'ci'
  args: string[]
}

const INSTALL_OPERATIONS = new Set(['install', 'i', 'add', 'ci'])
const BOOLEAN_FLAGS = new Set([
  '--save', '--save-prod', '--save-dev', '--save-optional', '--save-peer', '--save-exact',
  '-S', '-D', '-O', '-P', '-E', '--ignore-scripts', '--no-audit', '--no-fund', '--offline',
  '--prefer-offline', '--prefer-online', '--legacy-peer-deps', '--strict-peer-deps',
])
const VALUE_FLAGS: Record<string, RegExp> = {
  '--omit': /^(dev|optional|peer)$/,
  '--include': /^(prod|dev|optional|peer)$/,
}
const PACKAGE_NAME = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/i
const VERSION = /^[a-z0-9.*+_~^<>=| -]+$/i

function commandWords(command: string): string[] {
  if (command.length > 8192 || /[\r\n\x00-\x1f`$\\]/.test(command)) {
    throw new Error('Use one plain npm install or npm ci command without shell expansion.')
  }
  const words: string[] = []
  let word = ''
  let quote = ''
  for (const character of command.trim()) {
    if (quote) {
      if (character === quote) quote = ''
      else word += character
    } else if (character === '"' || character === "'") {
      quote = character
    } else if (/\s/.test(character)) {
      if (word) { words.push(word); word = '' }
    } else {
      if (/[;&|<>(){}]/.test(character)) {
        throw new Error('Run dependency installation separately from other shell commands.')
      }
      word += character
    }
  }
  if (quote) throw new Error('Dependency installation has an unterminated quote.')
  if (word) words.push(word)
  return words
}

export function isRegistryPackageName(value: string): boolean {
  return PACKAGE_NAME.test(value) && value.length <= 214
}

export function isRegistryVersion(value: string): boolean {
  if (!value || value.length > 256) return false
  if (value.startsWith('npm:')) return isRegistryPackageSpec(value.slice(4))
  return VERSION.test(value)
}

export function isRegistryPackageSpec(value: string): boolean {
  const separator = value.indexOf('@', value.startsWith('@') ? 1 : 0)
  const name = separator < 0 ? value : value.slice(0, separator)
  const version = separator < 0 ? null : value.slice(separator + 1)
  return isRegistryPackageName(name) && (version === null || isRegistryVersion(version))
}

export function parsePackageInstallCommand(command: string): PackageInstallCommand {
  const words = commandWords(command)
  if (words[0] !== 'npm' || !INSTALL_OPERATIONS.has(words[1])) {
    throw new Error(
      'Persistent dependency setup supports npm install and npm ci. '
      + 'Other package managers and package execution or scaffolding need a separately configured workflow.',
    )
  }
  const operation = words[1] === 'ci' ? 'ci' : 'install'
  const args = [operation]
  let packages = 0
  for (let index = 2; index < words.length; index++) {
    const word = words[index]
    if (BOOLEAN_FLAGS.has(word)) { args.push(word); continue }
    const separator = word.indexOf('=')
    const flag = separator < 0 ? word : word.slice(0, separator)
    const inline = separator < 0 ? undefined : word.slice(separator + 1)
    const rule = VALUE_FLAGS[flag]
    if (rule) {
      const value = inline ?? words[++index]
      if (!value || !rule.test(value)) throw new Error(`Unsupported ${flag} value for dependency setup.`)
      args.push(`${flag}=${value}`)
      continue
    }
    if (word.startsWith('-')) {
      throw new Error(`Dependency setup does not support ${word}; global, prefix and lifecycle overrides are refused.`)
    }
    if (operation === 'ci' || !isRegistryPackageSpec(word) || ++packages > 64) {
      throw new Error('Use registry package names with npm install; npm ci accepts no package arguments.')
    }
    args.push(word)
  }
  args.push('--ignore-scripts', '--no-audit', '--no-fund', '--install-links=false')
  return { manager: 'npm', operation, args }
}

export function packageInstallShellCommand(plan: PackageInstallCommand, stage: string): string {
  const quote = (value: string) => `'${value.replace(/'/g, "'\\''")}'`
  return [
    'npm', ...plan.args,
    '--registry=https://registry.npmjs.org/',
    `--cache=${stage}/.npm-cache`,
    `--userconfig=${stage}/.npm-userconfig`,
    `--globalconfig=${stage}/.npm-globalconfig`,
  ].map(quote).join(' ')
}
