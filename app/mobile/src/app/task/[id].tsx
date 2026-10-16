import { useState } from 'react'
import { KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native'
import { Redirect, router, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Body, Button, Field, Heading, Notice, layout, useColors } from '../../components/ui'
import { useSession } from '../../lib/session'
import { mobileThreadState, RESTRICTED_CAPABILITIES } from '../../lib/client'

export default function Task() {
  const { id } = useLocalSearchParams<{ id: string }>(); const c = useColors(); const session = useSession()
  const [text, setText] = useState(''); const [busy, setBusy] = useState<'send' | 'stop' | null>(null); const [error, setError] = useState<string | null>(null); const [notice, setNotice] = useState<string | null>(null); const [showChanges, setShowChanges] = useState(false)
  if (session.ready && !session.session) return <Redirect href="/" />
  const thread = session.host?.snapshot.threads.find((thread) => thread.id === id)
  const activeIds = session.host?.snapshot.activeThreadIds ?? []; const active = activeIds.includes(id)
  const state = thread ? mobileThreadState(thread, activeIds) : 'idle'
  const caps = session.host?.status?.capabilities ?? RESTRICTED_CAPABILITIES
  const engine = session.host?.status?.engine
  const localReady = thread?.providerId !== 'tensorfold' || engine?.state === 'ready' && engine.model === thread?.model
  const canSend = !session.error && !!thread && (active ? caps.taskGuide : caps.taskStart && localReady)
  const send = async () => {
    if (!session.client || busy) return
    setBusy('send'); setError(null); setNotice(null)
    try { await session.client.send(id, text, active, caps); setText(''); setNotice(active ? 'Instruction sent to the running worker.' : 'Message accepted by the host.'); await session.refresh() }
    catch (failure) { setError((failure as Error).message) }
    finally { setBusy(null) }
  }
  const stop = async () => {
    if (!session.client || busy) return
    setBusy('stop'); setError(null); setNotice(null)
    try { await session.client.stop(id, caps); setNotice('Stop requested; the host will preserve this task and its worktree.'); await session.refresh() }
    catch (failure) { setError((failure as Error).message) }
    finally { setBusy(null) }
  }
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><KeyboardAvoidingView style={layout.page} behavior={Platform.OS === 'ios' ? 'padding' : undefined}><ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={[layout.content, { paddingBottom: 36 }]}>
    <View style={layout.row}><Button secondary onPress={() => router.back()}>Back</Button>{active && <Button testID="task-stop" secondary danger disabled={!caps.taskStop || !!session.error} busy={busy === 'stop'} onPress={() => void stop()}>Stop</Button>}</View>
    {!thread ? <Notice>{session.syncing ? 'Loading this task from TensorFold.' : 'This task is unavailable; it may have been archived on the host.'}</Notice> : <>
      <View style={{ gap: 8 }}><Heading>{thread.title}</Heading><Body muted>{state === 'waiting' ? 'Waiting for review on the host' : state === 'running' ? 'Running on your Mac' : state === 'failed' ? 'The task needs attention' : state === 'finished' ? 'Finished; review on the host' : 'Ready to continue'}</Body>{thread.model && <Body muted>{thread.model}</Body>}</View>
      {(error || session.error) && <Notice error>{error || session.error}</Notice>}{notice && <Notice>{notice}</Notice>}
      <View style={[layout.card, { borderColor: c.line, backgroundColor: c.card, gap: 14 }]}><Heading small>{active ? 'Guide this worker' : 'Continue this task'}</Heading><Field testID="task-message" label={active ? 'Your instruction' : 'Your message'} value={text} onChangeText={setText} placeholder="Describe the next step" multiline maxLength={8_000} style={{ minHeight: 110 }} editable={!busy} /><Button testID="task-send" onPress={() => void send()} disabled={!canSend || !text.trim()} busy={busy === 'send'}>{active ? 'Send instruction' : 'Send message'}</Button>{!localReady && !active && <Body muted>Load the local model for this task in TensorFold before continuing from your phone.</Body>}{!(active ? caps.taskGuide : caps.taskStart) && <Body muted>This host session permits monitoring only; task controls are disabled.</Body>}</View>
      <View style={{ gap: 14 }}><View style={layout.row}><Heading small>{showChanges ? 'Changes' : 'Activity'}</Heading><Button secondary onPress={() => setShowChanges((show) => !show)}>{showChanges ? 'Show activity' : 'Review changes'}</Button></View>
        {showChanges && !caps.changesReview ? <Notice>Full paths and diffs stay on the desktop for this paired session; review the Changes pane for this worker in TensorFold.</Notice> : !caps.transcriptReview && <Notice>Task text and tool output stay private on the host; this device receives activity and run state.</Notice>}
        {(showChanges ? thread.timeline.filter((item) => item.diff?.length && caps.changesReview) : thread.timeline.slice(-40)).map((item) => <View key={item.id} style={{ borderTopWidth: 1, borderTopColor: c.line, paddingTop: 13, gap: 6 }}><Body style={{ fontWeight: '600' }}>{item.tool || item.kind.replace(/_/g, ' ')}{item.state ? `, ${item.state}` : ''}</Body>{caps.transcriptReview && item.text && <Body>{item.text}</Body>}{caps.transcriptReview && item.output && <Body muted>{item.output.slice(0, 6000)}</Body>}{showChanges && caps.changesReview && item.diff?.map((line, index) => <Body key={index} style={{ fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', fontSize: 13, color: line.type === 'add' ? c.positive : line.type === 'del' ? c.danger : c.ink2 }}>{line.type === 'add' ? '+' : line.type === 'del' ? '-' : ' '}{line.text}</Body>)}</View>)}
        {!thread.timeline.length && <Body muted>No activity yet; send a task after loading its model on the Mac.</Body>}
      </View>
    </>}
  </ScrollView></KeyboardAvoidingView></SafeAreaView>
}
