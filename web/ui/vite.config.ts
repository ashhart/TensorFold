import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built files land in the Python package, so `pip install` ships the page without Node.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../tensorfold_web/ui_dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8090', '/engine': 'http://127.0.0.1:8090' },
  },
})
