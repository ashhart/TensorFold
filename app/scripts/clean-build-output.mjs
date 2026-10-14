#!/usr/bin/env node
import { rm } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export async function cleanBuildOutputs(root) {
  await Promise.all([
    rm(path.join(root, 'dist'), { recursive: true, force: true }),
    rm(path.join(root, 'dist-electron'), { recursive: true, force: true }),
  ])
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  const root = fileURLToPath(new URL('..', import.meta.url))
  cleanBuildOutputs(root).catch((error) => {
    console.error(`[clean-build-output] ${error?.message || error}`)
    process.exitCode = 1
  })
}
