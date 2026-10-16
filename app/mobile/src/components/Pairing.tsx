import { useEffect, useState } from 'react'
import { KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native'
import { router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useSession } from '../lib/session'
import { parsePairingInput } from '../lib/client'
import { Body, Brand, Button, Field, Heading, Notice, layout, useColors } from './ui'

export function Pairing() {
  const c = useColors(); const session = useSession()
  const [host, setHost] = useState(''); const [credential, setCredential] = useState(''); const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (session.pairingLink) setCredential(session.pairingLink) }, [session.pairingLink])
  useEffect(() => {
    if (!/^https?:\/\//i.test(credential)) return
    try { setHost(parsePairingInput(credential, undefined, Platform.OS === 'android' && __DEV__).baseUrl) }
    catch { /* A partial pasted link remains editable until it is valid. */ }
  }, [credential])
  const pair = async () => {
    if (busy) return
    const enteredCredential = credential
    setBusy(true); setError(null)
    try {
      const parsed = parsePairingInput(credential, host, Platform.OS === 'android' && __DEV__)
      // Remove the consumed one-time value before the native form closes, preventing password-save heuristics.
      setCredential('')
      await session.pair(parsed.baseUrl, parsed.credential, parsed.certificatePin)
      router.replace('/')
    } catch (failure) { setCredential(enteredCredential); setError((failure as Error).message || 'Could not pair with TensorFold.') }
    finally { setBusy(false) }
  }
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><KeyboardAvoidingView style={layout.page} behavior={Platform.OS === 'ios' ? 'padding' : undefined}><ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={[layout.content, { paddingTop: 24, paddingBottom: 36 }]}>
    <Brand />
    <View style={{ gap: 10, marginTop: 28 }}><Heading>Connect to your Mac</Heading><Body muted>Follow workers and send instructions from your phone; your models keep running in TensorFold on the Mac.</Body></View>
    <View style={[layout.card, { backgroundColor: c.card, borderColor: c.line, gap: 18 }]}>
      <Heading small>Pair this device</Heading><Body muted>On your Mac, open Settings, Remote control, then Connect your phone; enable the private connection and scan its QR code.</Body>
      <Field testID="pair-host" label="Private host URL" value={host} onChangeText={setHost} autoCapitalize="none" autoCorrect={false} autoComplete="off" textContentType="none" importantForAutofill="no" keyboardType="url" placeholder="https://your-private-host" editable={!busy} />
      <Field testID="pair-code" label="Pairing link or one-time code" value={credential} onChangeText={setCredential} autoCapitalize="none" autoCorrect={false} autoComplete="off" textContentType="none" importantForAutofill="no" secureTextEntry editable={!busy} placeholder="Paste the link from TensorFold" />
      <Button testID="pair-submit" busy={busy} disabled={!credential.trim()} onPress={() => void pair()}>{busy ? 'Connecting' : 'Connect to TensorFold'}</Button>
      <Button secondary onPress={() => router.push('/scan')} disabled={busy}>Scan pairing QR code</Button>
    </View>
    {(error || session.error) && <Notice error>{error || session.error}</Notice>}
    <View style={{ gap: 8 }}><Heading small>Connect on your private network</Heading><Body muted>Keep your phone and Mac on the same private network, with TensorFold open and the Mac awake; the QR code verifies the encrypted host connection.</Body><Body muted>This app stores a revocable device token in secure phone storage; you can revoke it in TensorFold at any time.</Body></View>
  </ScrollView></KeyboardAvoidingView></SafeAreaView>
}
