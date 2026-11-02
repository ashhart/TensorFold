import path from 'node:path'
import { runEngineCli, parseVersion, parseCapabilities } from '../cli'
import { engineChildEnv } from '../engine-env'
import type { RuntimePackage } from './catalog'

export async function probeRuntime(dir: string, pkg: RuntimePackage): Promise<void> {
  const binary = { command: path.join(dir, pkg.entrypoint), args: [] }
  const env = engineChildEnv(process.env)
  const version = await runEngineCli(binary, ['--version'], env, 10_000)
  if (version.code !== 0 || parseVersion(version.stdout) !== pkg.version) throw new Error('Runtime version does not match its signed package.')
  const result = await runEngineCli(binary, ['capabilities', '--json'], env, 10_000)
  const raw = JSON.parse(result.stdout)
  const caps = parseCapabilities(result.stdout)
  if (result.code !== 0 || raw.schema !== pkg.protocol || caps.version !== pkg.version || !caps.backends.includes(pkg.backend) ||
    !Array.isArray(raw.optional_libraries) || !raw.optional_libraries.some((lib: { name?: string; found?: boolean }) => lib.name === 'TensorFold grammar' && lib.found === true)) throw new Error('Runtime protocol or backend is incompatible with this app.')
}
