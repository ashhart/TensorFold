import { createPrivateKey, X509Certificate } from 'node:crypto'
import { chmod, lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import path from 'node:path'

const run = promisify(execFile)

export interface MobileBridgeCertificate {
  cert: Buffer
  key: Buffer
  sha256: string
  expiresAt: number
}

export async function ensureMobilePrivateDirectory(directory: string): Promise<void> {
  await mkdir(directory, { recursive: true, mode: 0o700 })
  const info = await lstat(directory)
  if (!info.isDirectory() || info.isSymbolicLink() || process.getuid && info.uid !== process.getuid()) throw new Error('The mobile bridge profile directory is not privately owned.')
  await chmod(directory, 0o700)
}

async function privateFile(filename: string): Promise<Buffer> {
  const info = await lstat(filename)
  if (!info.isFile() || info.isSymbolicLink() || info.size > 64 * 1024 || process.getuid && info.uid !== process.getuid()) throw new Error('The mobile bridge certificate files are not privately owned.')
  await chmod(filename, 0o600)
  return readFile(filename)
}

function validate(cert: Buffer, key: Buffer, host: string): MobileBridgeCertificate {
  const certificate = new X509Certificate(cert), privateKey = createPrivateKey(key)
  if (!certificate.checkIP(host) || Date.parse(certificate.validFrom) > Date.now() || Date.parse(certificate.validTo) <= Date.now() + 7 * 86_400_000) throw new Error('The mobile bridge certificate needs renewal for this interface.')
  if (!certificate.checkPrivateKey(privateKey)) throw new Error('The mobile bridge certificate and key do not match.')
  return { cert, key, sha256: certificate.fingerprint256.replace(/:/g, '').toLowerCase(), expiresAt: Date.parse(certificate.validTo) }
}

/** The certificate/key stay in the private profile and never enter renderer state or a bundle. */
export async function loadMobileBridgeCertificate(profileRoot: string, host: string): Promise<MobileBridgeCertificate> {
  if (!/^(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})$/.test(host) || host.split('.').some(part => Number(part) > 255)) throw new Error('Choose a private IPv4 interface for the mobile certificate.')
  const directory = path.join(profileRoot, 'mobile-bridge')
  await ensureMobilePrivateDirectory(profileRoot)
  await ensureMobilePrivateDirectory(directory)
  const keyPath = path.join(directory, 'private-key.pem'), certPath = path.join(directory, 'certificate.pem')
  try { return validate(await privateFile(certPath), await privateFile(keyPath), host) }
  catch (failure) {
    if (failure instanceof Error && /privately owned/.test(failure.message)) throw failure
    // A different selected interface or an expired certificate requires a new pin and pairing.
  }
  const temporary = await mkdtemp(path.join(directory, '.certificate-'))
  try {
    const generatedKey = path.join(temporary, 'key.pem'), generatedCert = path.join(temporary, 'certificate.pem'), configPath = path.join(temporary, 'openssl.cnf')
    await writeFile(generatedKey, '', { mode: 0o600 })
    await writeFile(generatedCert, '', { mode: 0o600 })
    await writeFile(configPath, `[req]\nprompt = no\ndistinguished_name = dn\nx509_extensions = server\n[dn]\nCN = TensorFold private mobile bridge\n[server]\nbasicConstraints = critical,CA:FALSE\nkeyUsage = critical,digitalSignature,keyEncipherment\nextendedKeyUsage = serverAuth\nsubjectAltName = IP:${host}\n`, { mode: 0o600 })
    try {
      await run('/usr/bin/openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-sha256', '-nodes', '-days', '365', '-config', configPath, '-keyout', generatedKey, '-out', generatedCert], { timeout: 30_000, maxBuffer: 64 * 1024 })
    } catch { throw new Error('TensorFold could not create the private mobile certificate.') }
    const result = validate(await privateFile(generatedCert), await privateFile(generatedKey), host)
    // Validate the complete pair before replacing either of the old files.
    await rename(generatedKey, keyPath)
    await rename(generatedCert, certPath)
    return result
  } finally { await rm(temporary, { recursive: true, force: true }) }
}
