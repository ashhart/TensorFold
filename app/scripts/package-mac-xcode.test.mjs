import assert from 'node:assert/strict'
import { archiveInfoPlist, isStaplerTicketPending, isXcodeNotarizationProcessing, xcodeExportOptionsPlist } from './package-mac-xcode.mjs'

const signer = { identity: 'Developer ID Application: Example & Team (ABC1234567)', teamId: 'ABC1234567' }
const metadata = { productName: 'Studio & Test', appId: 'dev.example.studio', version: '1.0.0-beta.2', arch: 'arm64' }
const archive = archiveInfoPlist(metadata, signer, new Date('2026-10-12T00:00:00Z'))
assert.match(archive, /Applications\/Studio &amp; Test\.app/)
assert.match(archive, /Developer ID Application: Example &amp; Team \(ABC1234567\)/)
assert.match(archive, /<key>ArchiveVersion<\/key><integer>2<\/integer>/)
assert.match(xcodeExportOptionsPlist(signer), /<key>destination<\/key><string>upload<\/string>/)
assert.match(xcodeExportOptionsPlist(signer), /<key>method<\/key><string>developer-id<\/string>/)
assert.throws(() => archiveInfoPlist({ ...metadata, productName: '../escape' }, signer), /safe artifact/)
assert.throws(() => archiveInfoPlist(metadata, { ...signer, teamId: 'ZZZ1234567' }), /exact pinned/)
assert.equal(isXcodeNotarizationProcessing(65, 'error: Archive "Studio.xcarchive" is processing and not ready for distribution.'), true)
assert.equal(isXcodeNotarizationProcessing(65, 'error: Archive "Studio.xcarchive" has been rejected.'), false)
assert.equal(isXcodeNotarizationProcessing(1, 'error: Archive "Studio.xcarchive" is processing and not ready for distribution.'), false)
assert.equal(isXcodeNotarizationProcessing(65, 'Authentication failed.'), false)
assert.equal(isStaplerTicketPending(65, 'CloudKit query failed due to "Record not found".'), true)
assert.equal(isStaplerTicketPending(65, 'The ticket does not match.'), false)
console.log('Xcode archive metadata and pinned signer tests passed')
