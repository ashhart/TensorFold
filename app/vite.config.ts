import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import electron from 'vite-plugin-electron/simple'
import path from 'node:path'

export default defineConfig(({ command }) => ({
  resolve: {
    alias: { '@': path.resolve(process.cwd(), 'src') },
  },
  server: {
    host: '127.0.0.1',
    port: process.env.PORT ? Number(process.env.PORT) : 5173,
  },
  plugins: [
    {
      name: 'vontra-production-csp-meta',
      transformIndexHtml(html) {
        return command === 'serve' ? html.replace(/\s*<meta data-vontra-production-csp[^>]*>/, '') : html
      },
    },
    react(),
    tailwindcss(),
    ...(process.env.VONTRA_NO_ELECTRON ? [] : [electron({
      main: {
        entry: { main: 'electron/main.ts', 'kokoro-worker': 'electron/voice/kokoro-worker.ts' },
        vite: {
          build: {
            outDir: 'dist-electron',
            rollupOptions: {
              // @whiskeysockets/baileys (WhatsApp bridge) can't be bundled by rollup —
              // it was being emitted as an empty 110B stub, so loadBaileys() resolved to
              // {} and pairing silently did nothing. Externalize it so the main bundle
              // require()s the real package from the shipped node_modules at runtime.
              external: ['electron', 'node:sqlite', '@whiskeysockets/baileys', '@lydell/node-pty', 'kokoro-js', '@huggingface/transformers'],
              output: { format: 'cjs', entryFileNames: '[name].js' },
            },
          },
        },
      },
      preload: {
        input: 'electron/preload.ts',
        vite: {
          build: {
            outDir: 'dist-electron',
            rollupOptions: {
              external: ['electron'],
              output: { format: 'cjs', entryFileNames: 'preload.js' },
            },
          },
        },
      },
    })]),
  ],
  build: {
    outDir: 'dist',
  },
}))
