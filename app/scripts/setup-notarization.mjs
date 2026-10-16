#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { assertProductionMacSigner, readReleaseTrustConfig } from './update-signing-trust.mjs'

if (!process.stdin.isTTY) throw new Error('Run release:credentials in an interactive local terminal; enter Apple credentials there.')
const trust = await readReleaseTrustConfig()
const signer = await assertProductionMacSigner(undefined, trust)
// Apple's tool prompts locally; this script never reads credentials or passes them in arguments.
const result = spawnSync('xcrun', ['notarytool', 'store-credentials', 'tensorfold-notary', '--team-id', signer.teamId], { stdio: 'inherit' })
if (result.error) throw result.error
process.exitCode = result.status ?? 1
