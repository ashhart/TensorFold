// Attach a local donation node to a relay, so renters can reach it over HTTPS.
//
//   node node_modules/jiti/lib/jiti-cli.mjs scripts/donation-relay-attach.mjs \
//     --relay https://relay.example --id donate-abc123 --key <donor-key> --local 8020
//
// The local node keeps enforcing the card token, the window and the caps; this
// process only carries frames between the relay and 127.0.0.1.
import { randomBytes } from 'node:crypto'
import { attachDonationRelay } from '../electron/donation-relay-client.ts'

const args = new Map()
const argv = process.argv.slice(2)
for (let index = 0; index < argv.length; index += 1) {
  const flag = argv[index]
  if (!flag.startsWith('--')) continue
  args.set(flag.slice(2), argv[index + 1]?.startsWith('--') ? 'true' : (argv[++index] ?? 'true'))
}

const relay = args.get('relay')
const cardId = args.get('id')
const localPort = Number(args.get('local') ?? 8020)
const donorKey = args.get('key') ?? randomBytes(24).toString('hex')

if (!relay || !cardId) {
  process.stderr.write('usage: donation-relay-attach.mjs --relay <https://relay> --id <card-id> [--key <donor-key>] [--local <port>]\n')
  process.exit(2)
}
if (!args.has('key')) process.stdout.write(`donor key (keep it, the relay binds it to ${cardId}): ${donorKey}\n`)

const attachment = attachDonationRelay({
  relayUrl: relay,
  cardId,
  donorKey,
  localOrigin: `http://127.0.0.1:${localPort}`,
  log: (line) => process.stdout.write(`${new Date().toISOString()} ${line}\n`),
})

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    attachment.close()
    process.exit(0)
  })
}
