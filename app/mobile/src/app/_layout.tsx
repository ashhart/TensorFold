import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import { SessionProvider } from '../lib/session'
import { WatchBridgeProvider } from '../lib/watch-bridge'
import { useColors } from '../components/ui'

export default function RootLayout() {
  const c = useColors()
  return <SessionProvider><WatchBridgeProvider><StatusBar style="auto" /><Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: c.bg }, animation: 'slide_from_right' }} /></WatchBridgeProvider></SessionProvider>
}
