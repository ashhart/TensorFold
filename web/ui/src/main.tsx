import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { redeemFromAddress } from './api'
import './styles.css'

// a one-time link opened here signs this browser in before anything asks TensorFold Web
void redeemFromAddress().finally(() => createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
))
