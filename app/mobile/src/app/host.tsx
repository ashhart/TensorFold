import { ScrollView, View } from 'react-native'
import { Redirect, router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useSession } from '../lib/session'
import { Body, Button, Heading, Notice, layout, useColors } from '../components/ui'

export default function Host() {
  const c = useColors(); const session = useSession(); const engine = session.host?.status?.engine; const caps = session.host?.status?.capabilities
  if (session.ready && !session.session) return <Redirect href="/" />
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><ScrollView contentContainerStyle={[layout.content, { paddingBottom: 36 }]}><Button secondary onPress={() => router.back()}>Back</Button><Heading>Your host</Heading><View style={[layout.card, { borderColor: c.line, backgroundColor: c.card }]}><Heading small>{session.host?.snapshot.serverName || 'TensorFold'}</Heading><Body muted>{session.session?.baseUrl}</Body><Body muted>{session.error ? 'Connection interrupted' : session.lastSynced ? `Last synced ${new Date(session.lastSynced).toLocaleTimeString()}` : 'Connecting'}</Body>{session.host?.snapshot.appVersion && <Body muted>TensorFold {session.host.snapshot.appVersion}</Body>}<Body muted>{caps?.taskStart ? 'Paired device; task controls enabled' : 'Paired device; monitoring only'}</Body></View>
    {session.error && <Notice error>{session.error}</Notice>}
    <View style={{ gap: 10 }}><Heading small>Model on your Mac</Heading><Body>{engine?.model || 'No model loaded'}</Body><Body muted>{engine?.hardware?.chip || 'Hardware details unavailable'}</Body>{!!engine?.totalMemoryBytes && <Body muted>{Math.round(engine.totalMemoryBytes / 1024 ** 3)} GiB installed memory; {Math.round(engine.memoryLimitBytes / 1024 ** 3)} GiB engine budget</Body>}<Body muted>Model downloads and launch settings are managed in TensorFold on the host.</Body></View>
    <View style={{ gap: 10 }}><Heading small>Device permissions</Heading><Body muted>{caps?.taskStart ? 'Send and guide tasks on the host.' : 'Monitor tasks only.'}</Body><Body muted>{caps?.changesReview ? 'Review task transcripts and changes.' : 'Private task text, approvals and full diffs remain on the desktop.'}</Body><Body muted>This phone cannot manage device sessions, open a shell, or change engine settings.</Body></View>
    <Button secondary onPress={() => void session.refresh()} busy={session.syncing}>Refresh connection</Button><Button secondary danger onPress={() => void session.forget().then(() => router.replace('/'))}>Forget this host</Button><Body muted>Forgetting removes the token from this phone; revoke the device in TensorFold to end its host session.</Body>
  </ScrollView></SafeAreaView>
}
