// Pure core for provider-CLI version advisories (VST-037).
// Given the installed version string a CLI reports (`--version`) and the latest known
// version, decide whether an update is available and surface the command to run. Kept pure
// so the readiness doctor + settings can render advisories without a network call in the
// hot path (the host fetches "latest" out-of-band and passes it in).

// Status meanings: 'current' installed>=latest · 'update_available' installed<latest ·
// 'ahead' installed>latest (dev/prerelease, informational) · 'unknown' unparseable or latest
// not known yet. Types live in ./types/cli so the renderer can consume advisories as data.
import type { VersionAdvisory, VersionAdvisoryStatus } from './types/cli'

export type { VersionAdvisory, VersionAdvisoryStatus }

export interface Semver {
  major: number
  minor: number
  patch: number
  // Prerelease tag (the `-beta.1` in `1.2.3-beta.1`); '' when none. A version WITH a
  // prerelease sorts BEFORE the same version without one (semver §11).
  prerelease: string
}

// CLIs report versions in many shapes: "0.142.5", "codex-cli 0.142.5", "2.1.170 (Claude Code)",
// "v1.14.20", "gpt-5 pi 0.80.3". Pull the FIRST x.y(.z) run and any trailing -prerelease.
export function parseSemver(raw: string | null | undefined): Semver | null {
  if (!raw) return null
  const match = String(raw).match(/(\d+)\.(\d+)(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?/)
  if (!match) return null
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: match[3] ? Number(match[3]) : 0,
    prerelease: match[4] ?? '',
  }
}

function comparePrerelease(a: string, b: string): number {
  // No prerelease outranks a prerelease (1.2.3 > 1.2.3-rc.1).
  if (a === b) return 0
  if (!a) return 1
  if (!b) return -1
  const pa = a.split('.')
  const pb = b.split('.')
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i]
    const y = pb[i]
    if (x === undefined) return -1
    if (y === undefined) return 1
    const nx = /^\d+$/.test(x)
    const ny = /^\d+$/.test(y)
    if (nx && ny) {
      const d = Number(x) - Number(y)
      if (d !== 0) return d < 0 ? -1 : 1
    } else if (nx !== ny) {
      // Numeric identifiers always have lower precedence than alphanumeric (semver §11.4.3).
      return nx ? -1 : 1
    } else if (x !== y) {
      return x < y ? -1 : 1
    }
  }
  return 0
}

// -1 if a<b, 0 if equal, 1 if a>b. Nulls sort last (unknown is "smallest").
export function compareSemver(a: Semver | null, b: Semver | null): number {
  if (!a && !b) return 0
  if (!a) return -1
  if (!b) return 1
  if (a.major !== b.major) return a.major < b.major ? -1 : 1
  if (a.minor !== b.minor) return a.minor < b.minor ? -1 : 1
  if (a.patch !== b.patch) return a.patch < b.patch ? -1 : 1
  return comparePrerelease(a.prerelease, b.prerelease)
}

export function formatSemver(v: Semver): string {
  const base = `${v.major}.${v.minor}.${v.patch}`
  return v.prerelease ? `${base}-${v.prerelease}` : base
}

export interface VersionAdvisoryInput {
  name: string
  installedVersion?: string | null
  latestVersion?: string | null
  updateCommand?: string | null
  // False when the binary isn't installed at all — advisory is not applicable then.
  installed?: boolean
}

export function versionAdvisory(input: VersionAdvisoryInput): VersionAdvisory {
  const installed = parseSemver(input.installedVersion)
  const latest = parseSemver(input.latestVersion)
  const currentVersion = installed ? formatSemver(installed) : (input.installedVersion?.trim() || null)
  const latestVersion = latest ? formatSemver(latest) : (input.latestVersion?.trim() || null)
  const updateCommand = input.updateCommand?.trim() || null

  if (input.installed === false) {
    return { status: 'unknown', currentVersion, latestVersion, updateCommand, canUpdate: false, message: `${input.name} is not installed.` }
  }
  if (!installed || !latest) {
    return {
      status: 'unknown',
      currentVersion,
      latestVersion,
      updateCommand,
      canUpdate: false,
      message: !latest ? `${input.name} update status unknown (latest version not checked yet).` : `${input.name} version could not be parsed.`,
    }
  }

  const cmp = compareSemver(installed, latest)
  if (cmp < 0) {
    return {
      status: 'update_available',
      currentVersion,
      latestVersion,
      updateCommand,
      canUpdate: Boolean(updateCommand),
      message: `${input.name} ${latestVersion} is available (you have ${currentVersion}).${updateCommand ? ` Update with \`${updateCommand}\`.` : ''}`,
    }
  }
  if (cmp > 0) {
    return {
      status: 'ahead',
      currentVersion,
      latestVersion,
      updateCommand,
      canUpdate: false,
      message: `${input.name} ${currentVersion} is ahead of the latest published ${latestVersion} (dev/prerelease build).`,
    }
  }
  return {
    status: 'current',
    currentVersion,
    latestVersion,
    updateCommand,
    canUpdate: false,
    message: `${input.name} is up to date (${currentVersion}).`,
  }
}
