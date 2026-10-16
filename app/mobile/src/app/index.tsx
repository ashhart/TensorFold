import { ActivityIndicator, FlatList, Pressable, RefreshControl, View } from 'react-native'
import { useState } from 'react'
import { router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Pairing } from '../components/Pairing'
import { Body, Brand, Button, Heading, Notice, layout, useColors } from '../components/ui'
import { useSession } from '../lib/session'
import { mobileThreadState } from '../lib/client'

export default function Workspace() {
  const c = useColors(); const session = useSession(); const [filter, setFilter] = useState<'task' | 'chat'>('task')
  if (!session.ready) return <SafeAreaView style={[layout.page, { backgroundColor: c.bg, alignItems: 'center', justifyContent: 'center' }]}><ActivityIndicator color={c.ink} /><Body muted>Restoring your connection</Body></SafeAreaView>
  if (!session.session) return <Pairing />
  const host = session.host; const snapshot = host?.snapshot; const engine = host?.status?.engine
  const threads = (snapshot?.threads ?? []).filter((thread) => thread.kind === filter).sort((a, b) => Number(snapshot!.activeThreadIds.includes(b.id)) - Number(snapshot!.activeThreadIds.includes(a.id)) || b.updatedAt - a.updatedAt)
  const running = snapshot?.activeThreadIds.length ?? 0
  const rate = engine?.state === 'ready' && engine.performance?.model === engine.model ? engine.performance.decodeTokensPerSecond : null
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><FlatList data={threads} keyExtractor={(thread) => thread.id} contentContainerStyle={[layout.content, { paddingBottom: 32 }]} refreshControl={<RefreshControl refreshing={session.syncing} onRefresh={() => void session.refresh()} tintColor={c.ink} />} ListHeaderComponent={<View style={{ gap: 22, marginBottom: 14 }}>
    <View style={layout.row}><Brand /><Button secondary onPress={() => router.push('/host')}>Host</Button></View>
    <View style={{ gap: 6 }}><Heading>{snapshot?.serverName || 'Your workspace'}</Heading><Body muted>{session.error ? 'Connection interrupted; showing the last workspace snapshot.' : snapshot ? `${running} ${running === 1 ? 'run' : 'runs'} active on your Mac` : 'Connecting to TensorFold'}</Body></View>
    {session.error && <View style={{ gap: 8 }}><Notice error>{session.error}</Notice><Button secondary onPress={() => void session.refresh()} busy={session.syncing}>Reconnect</Button></View>}
    <View style={[layout.card, { backgroundColor: c.card, borderColor: c.line }]}><View style={layout.row}><Heading small>Local model</Heading><Body muted>{engine?.state === 'ready' ? 'Ready' : engine?.state === 'starting' ? 'Loading' : engine?.state === 'failed' ? 'Needs attention' : engine ? 'Stopped' : 'Status unavailable'}</Body></View><Body>{engine?.model || 'Load a model in TensorFold on your Mac'}</Body>{engine?.hardware?.chip && <Body muted>{engine.hardware.chip}</Body>}{rate != null && <Body muted>Last completed request {rate.toFixed(1)} tokens/s</Body>}<Body muted>The phone controls your workspace; inference stays on the host.</Body></View>
    {!!snapshot?.work?.approvals?.pending && <Notice>{snapshot.work.approvals.pending} {snapshot.work.approvals.pending === 1 ? 'approval needs' : 'approvals need'} review on the host; paired sessions do not receive the private request details.</Notice>}
    <View accessibilityRole="tablist" style={[layout.row, { justifyContent: 'flex-start' }]}>{(['task', 'chat'] as const).map((kind) => <Pressable key={kind} accessibilityRole="tab" accessibilityState={{ selected: filter === kind }} onPress={() => setFilter(kind)} style={{ minHeight: 44, paddingHorizontal: 18, justifyContent: 'center', borderRadius: 10, backgroundColor: filter === kind ? c.card2 : c.bg }}><Body style={{ fontWeight: filter === kind ? '600' : '400' }}>{kind === 'task' ? 'Workers and tasks' : 'Chats'}</Body></Pressable>)}</View>
  </View>} ListEmptyComponent={<View style={{ paddingVertical: 28, gap: 10 }}><Heading small>{snapshot ? filter === 'task' ? 'No workers yet' : 'No chats yet' : 'Waiting for the host'}</Heading><Body muted>{snapshot ? 'Create a worker or chat in TensorFold on your Mac; it will appear here when the host syncs.' : 'Keep TensorFold running and your private connection available.'}</Body></View>} renderItem={({ item }) => {
    const state = mobileThreadState(item, snapshot?.activeThreadIds ?? [])
    return <Pressable testID={`task-${item.id}`} accessibilityRole="button" accessibilityLabel={`${item.title}, ${state}`} onPress={() => router.push({ pathname: '/task/[id]', params: { id: item.id } })} style={({ pressed }) => ({ borderTopWidth: 1, borderTopColor: c.line, paddingVertical: 18, gap: 6, opacity: pressed ? 0.6 : 1 })}><View style={layout.row}><Body style={{ fontWeight: '600', flex: 1 }}>{item.title}</Body><Body style={{ color: state === 'failed' ? c.danger : c.ink2 }}>{state === 'waiting' ? 'Needs you' : state === 'running' ? 'Running' : state === 'failed' ? 'Failed' : state === 'finished' ? 'Finished' : 'Ready'}</Body></View><Body muted>{snapshot?.projects.find((project) => project.id === item.projectId)?.name || (item.kind === 'task' ? 'Task' : 'Chat')}</Body>{item.model && <Body muted style={{ fontSize: 13 }}>{item.model}</Body>}</Pressable>
  }} /></SafeAreaView>
}
