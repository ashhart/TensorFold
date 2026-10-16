import { useEffect, useState } from 'react'
import { Linking, View } from 'react-native'
import { CameraView, useCameraPermissions } from 'expo-camera'
import { router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useSession } from '../lib/session'
import { parsePairingInput } from '../lib/client'
import { Body, Button, Heading, Notice, layout, useColors } from '../components/ui'

export default function Scan() {
  const c = useColors(); const session = useSession(); const [permission, request] = useCameraPermissions(); const [error, setError] = useState<string | null>(null); const [scanned, setScanned] = useState(false)
  const [available, setAvailable] = useState<boolean | null>(null)
  useEffect(() => {
    let alive = true
    void CameraView.isAvailableAsync().then(value => { if (alive) setAvailable(value) }).catch(() => { if (alive) setAvailable(false) })
    return () => { alive = false }
  }, [])
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><View style={[layout.content, { flex: 1 }]}><Button secondary onPress={() => router.back()}>Back</Button><Heading>Scan the TensorFold QR code</Heading><Body muted>Use the pairing link generated on your Mac; the host address and code will fill the connection form.</Body>
    {available === null ? <Body muted>Checking camera availability…</Body> : !available ? <Notice>There is no camera available on this device; go back and paste the pairing link from TensorFold.</Notice> : !permission?.granted ? <View style={{ gap: 14 }}><Notice>{permission?.canAskAgain === false ? 'Camera access is disabled; open device settings to enable it, or go back and paste the pairing link.' : 'Camera access is used only to scan the pairing QR code.'}</Notice><Button onPress={() => { if (permission?.canAskAgain === false) void Linking.openSettings(); else void request() }}>{permission?.canAskAgain === false ? 'Open device settings' : 'Allow camera'}</Button></View> : <CameraView style={{ flex: 1, minHeight: 240, borderRadius: 16 }} barcodeScannerSettings={{ barcodeTypes: ['qr'] }} onMountError={({ message }) => { setAvailable(false); setError(message) }} onBarcodeScanned={scanned ? undefined : ({ data }) => {
      try { parsePairingInput(data); setScanned(true); session.setPairingLink(data); router.back() } catch (failure) { setError((failure as Error).message); setScanned(true) }
    }} />}
    {error && <><Notice error>{error}</Notice><Button secondary onPress={() => { setError(null); setScanned(false) }}>Scan again</Button></>}
  </View></SafeAreaView>
}
