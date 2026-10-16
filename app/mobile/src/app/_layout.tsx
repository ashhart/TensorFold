import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import { SessionProvider } from '../lib/session'
import { useColors } from '../components/ui'

export default function RootLayout() {
  const c = useColors()
  return <SessionProvider><StatusBar style="auto" /><Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: c.bg }, animation: 'slide_from_right' }} /></SessionProvider>
}
