import { generateKeyPairSync, randomBytes, sign } from 'node:crypto'

// Ephemeral self-signed test material, generated per test and never shipped as a credential fixture.
export function ephemeralTls(): { cert: string; key: string } {
  const length = (value: number) => value < 128 ? Buffer.from([value]) : (() => { const bytes: number[] = []; while (value) { bytes.unshift(value & 255); value >>>= 8 }; return Buffer.from([0x80 | bytes.length, ...bytes]) })()
  const der = (tag: number, ...parts: Buffer[]) => { const value = Buffer.concat(parts); return Buffer.concat([Buffer.from([tag]), length(value.length), value]) }
  const sequence = (...parts: Buffer[]) => der(0x30, ...parts)
  const oid = (parts: number[]) => {
    const bytes = [parts[0] * 40 + parts[1]]
    for (let value of parts.slice(2)) { const encoded = [value & 127]; while ((value = Math.floor(value / 128)) > 0) encoded.unshift((value & 127) | 128); bytes.push(...encoded) }
    return der(6, Buffer.from(bytes))
  }
  const algorithm = sequence(oid([1, 2, 840, 10045, 4, 3, 2]))
  const name = sequence(der(0x31, sequence(oid([2, 5, 4, 3]), der(0x0c, Buffer.from('localhost')))))
  const time = (date: Date) => der(0x17, Buffer.from(date.toISOString().replace(/[-:T]/g, '').slice(2, 14) + 'Z'))
  const pair = generateKeyPairSync('ec', { namedCurve: 'prime256v1' })
  const validity = sequence(time(new Date(Date.now() - 86_400_000)), time(new Date(Date.now() + 86_400_000)))
  const san = sequence(oid([2, 5, 29, 17]), der(4, sequence(der(0x82, Buffer.from('localhost')), der(0x87, Buffer.from([127, 0, 0, 1])))))
  const constraints = sequence(oid([2, 5, 29, 19]), der(1, Buffer.from([255])), der(4, sequence(der(1, Buffer.from([255])))))
  // A DER INTEGER must be positive and minimal: a first byte of 0 (about one serial in 256) fails to parse
  const serial = randomBytes(16); serial[0] = 0x40 | (serial[0] & 0x3f)
  const tbs = sequence(der(0xa0, der(2, Buffer.from([2]))), der(2, serial), algorithm, name, validity, name, pair.publicKey.export({ type: 'spki', format: 'der' }) as Buffer, der(0xa3, sequence(san, constraints)))
  const certificate = sequence(tbs, algorithm, der(3, Buffer.from([0]), sign('sha256', tbs, pair.privateKey)))
  return { cert: '-----BEGIN CERTIFICATE-----\n' + certificate.toString('base64').match(/.{1,64}/g)!.join('\n') + '\n-----END CERTIFICATE-----\n', key: pair.privateKey.export({ type: 'pkcs8', format: 'pem' }) as string }
}
