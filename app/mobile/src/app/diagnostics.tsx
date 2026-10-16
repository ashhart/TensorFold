import { useState } from 'react'
import { Image, ScrollView, View } from 'react-native'
import { scanFromURLAsync } from 'expo-camera'
import { requireOptionalNativeModule } from 'expo'
import { Redirect, router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Body, Button, Heading, Notice, layout, useColors } from '../components/ui'
import fixtures from '../fixtures/pinning.json'
import qrFixture from '../fixtures/pairing-qr.json'
import { parsePairingInput } from '../lib/client'

interface TestModule { testCertificate(certificate: string, host: string, pin: string): Promise<{ accepted: boolean; reason: string }> }
export default function Diagnostics() {
  const c = useColors(); const [results, setResults] = useState<{ name: string; passed: boolean; reason: string }[]>([]); const [busy, setBusy] = useState(false)
  if (!__DEV__) return <Redirect href="/" />
  const run = async () => {
    setBusy(true)
    try {
      const native = requireOptionalNativeModule<TestModule>('TensorFoldPinnedNetwork')
      if (!native?.testCertificate) throw new Error('Use the iOS native debug build for these checks.')
      const next = []
      for (const fixture of fixtures) {
        const result = await native.testCertificate(fixture.certificate, fixture.host, fixture.pin)
        next.push({ name: fixture.name, passed: result.accepted === fixture.expected, reason: result.reason })
      }
      setResults(next)
    } catch (error) { setResults([{ name: 'Native certificate checks', passed: false, reason: (error as Error).message }]) }
    finally { setBusy(false) }
  }
  const runQr = async () => {
    setBusy(true)
    try {
      const image = Image.resolveAssetSource(require('../fixtures/pairing-qr.png'))
      const codes = await scanFromURLAsync(image.uri, ['qr'])
      if (codes[0]?.data !== qrFixture.link) throw new Error('The native scanner did not decode the public QR fixture exactly.')
      const parsed = parsePairingInput(codes[0].data)
      setResults([{ name: 'Native QR decoding', passed: parsed.certificatePin === '0'.repeat(64), reason: 'Native image scan decoded the host, one-time-code field and certificate pin; no live credential was used.' }])
    } catch (failure) { setResults([{ name: 'Native QR decoding', passed: false, reason: (failure as Error).message }]) }
    finally { setBusy(false) }
  }
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><ScrollView contentContainerStyle={layout.content}><Button secondary onPress={() => router.replace('/')}>Back</Button><Heading>Native connection checks</Heading><Body muted>Development-only public fixtures exercise native certificate and QR decoders.</Body><Button testID="run-pinning-checks" onPress={() => void run()} busy={busy}>Run certificate checks</Button><Button testID="run-qr-checks" secondary onPress={() => void runQr()} busy={busy}>Decode public QR fixture</Button>{results.map((result) => <View key={result.name} style={{ gap: 7 }}><Heading small>{result.name}</Heading><Notice error={!result.passed}>{result.passed ? 'Passed' : 'Failed'}; {result.reason}</Notice></View>)}</ScrollView></SafeAreaView>
}
