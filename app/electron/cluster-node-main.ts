import { privateRead, publicMessage, validateNodeConfig } from './cluster-node/config'
import { createClusterNodeServer } from './cluster-node/server'

/** Standalone Node 22 entry. It neither imports Electron nor alters OS services or driver policy. */
export async function clusterNodeMain(args = process.argv.slice(2)): Promise<void> {
  if (args.length === 1 && (args[0] === '--help' || args[0] === '-h')) { process.stdout.write('Usage: node cluster-node.js --config /absolute/private/config.json\n'); return }
  if (args.length !== 2 || args[0] !== '--config' || !args[1].startsWith('/')) throw new Error('Use --config with one private absolute configuration file.')
  if (Number(process.versions.node.split('.')[0]) !== 22) throw new Error('The headless node requires Node.js 22.')
  const config = validateNodeConfig(JSON.parse((await privateRead(args[1], 64 * 1024)).toString('utf8')))
  const node = await createClusterNodeServer({ config, diagnostic: message => process.stderr.write(message + '\n') })
  let closing = false
  const stop = () => { if (closing) return; closing = true; void node.close().then(() => { process.exitCode = 0 }, () => { process.stderr.write('The headless node could not confirm physical teardown; review its owned engine before restarting.\n'); process.exitCode = 1 }) }
  process.once('SIGINT', stop); process.once('SIGTERM', stop)
  try { const address = await node.listen(); if (!closing) process.stdout.write(JSON.stringify({ protocol: 1, listening: true, port: address.port, tls: !!config.tls, advertised: config.advertise === true }) + '\n') }
  catch (error) { await node.close(); throw error }
}
if (typeof require !== 'undefined' && require.main === module) void clusterNodeMain().catch(error => { process.stderr.write(publicMessage(error) + '\n'); process.exitCode = 1 })
