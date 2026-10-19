import './lib/trusted-types'
import { installBootGuard } from './lib/boot-guard'

// First: make a failed or stalled boot visible instead of leaving the static splash forever.
installBootGuard(window, document)
import React from 'react'
import { createRoot } from 'react-dom/client'
import { preparePairedSession } from './lib/remote-pairing'
import './styles/index.css'
import { prepareBrowserDemo } from './lib/browser-demo-bootstrap'

// Boot order matters (VST-035): the paired-session decision must land BEFORE the app's
// module graph loads, because src/lib/ipc.ts picks its transport synchronously at import
// time (desktop preload → paired /v1 transport → browser demo mock). Hence the dynamic
// imports below.
async function boot() {
  const root = createRoot(document.getElementById('root')!)
  const mode = await preparePairedSession()
  if (mode === 'needs-pairing') {
    const { PairingGate } = await import('./components/PairingGate')
    root.render(
      <React.StrictMode>
        <PairingGate />
      </React.StrictMode>,
    )
    return
  }
  if (mode === 'demo') await prepareBrowserDemo()
  const { default: App } = await import('./App')
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

void boot()
