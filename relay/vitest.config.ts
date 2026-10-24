import { cloudflareTest } from '@cloudflare/vitest-plugin'
import { defineConfig } from 'vitest/config'

// Runs the tests inside the Workers runtime with the Durable Object from wrangler.toml.
export default defineConfig({
  plugins: [cloudflareTest({ wrangler: { configPath: './wrangler.toml' } })],
  test: { include: ['test/**/*.test.ts'] },
})
