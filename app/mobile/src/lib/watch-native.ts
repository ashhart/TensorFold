import { Platform } from 'react-native'
import { requireOptionalNativeModule } from 'expo'
import { createWatchBackgroundController, type BackgroundWatchPort } from './watch-background-controller'

const bridge = Platform.OS === 'ios' ? requireOptionalNativeModule<BackgroundWatchPort>('TensorFoldWatchConnectivity') : null
const supported = bridge && typeof bridge.configureBackground === 'function' && typeof bridge.backgroundStatus === 'function' ? bridge : null
export const backgroundWatch = createWatchBackgroundController(supported)

/** Explicit account/consent changes revoke native access even if React later unmounts. */
export async function disableBackgroundWatch(): Promise<void> { await backgroundWatch.disable() }
