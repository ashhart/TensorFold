import { createHash } from 'node:crypto'
import type { BackupRecord, WorkspaceBackupAttachment } from '../../../../shared/workspace-backup'
import type { Attachment, Thread } from '../../../agent/types'
import { backupDigest, MAX_BLOB_BYTES } from '../../../workspace-backup/archive'

export interface ManagedAttachmentSnapshot { attachments: WorkspaceBackupAttachment[]; blobs: Map<string, Uint8Array> }

/** Read only bytes retained in the chosen conversations; never follow original picker paths. */
export function captureManagedAttachments(threads: Thread[], existing: ManagedAttachmentSnapshot): ManagedAttachmentSnapshot {
  const descriptors = new Map(existing.attachments.map((item) => [item.id, item])), blobs = new Map(existing.blobs)
  const retain = (bytes: Uint8Array) => {
    if (bytes.byteLength > MAX_BLOB_BYTES) throw new Error('A managed attachment exceeds the backup file limit.')
    const id = backupDigest(bytes); blobs.set(id, bytes); return id
  }
  const imageBytes = (url: unknown): Uint8Array | null => {
    if (typeof url !== 'string' || url.length > Math.ceil(MAX_BLOB_BYTES / 3) * 4 + 100) return null
    const match = /^data:image\/[a-z0-9.+-]+;base64,([A-Za-z0-9+/]*={0,2})$/iu.exec(url)
    if (!match || match[1].length % 4) return null
    const bytes = Buffer.from(match[1], 'base64')
    if (bytes.toString('base64') !== match[1]) return null
    return bytes
  }
  for (const thread of threads) {
    const images = new Map<string, string>()
    const visitTimeline = (timeline: Thread['timeline']) => {
      for (const item of timeline ?? []) for (const attachment of item.attachments ?? []) {
        const value = attachment as Attachment & { managedBackupRef?: string }
        const id = value.managedBackupRef ?? createHash('sha256').update(`${thread.id}\0${item.id}\0${value.id}`).digest('hex')
        const current = descriptors.get(id)
        const original = imageBytes(value.dataUrl)
        const parsed = value.text ? Buffer.from(value.text, 'utf8') : null
        descriptors.set(id, { id, name: (value.name || 'Attachment.bin').replace(/[\\/:\u0000-\u001f\u007f]/gu, '_').slice(0, 240), mediaType: value.mime || 'application/octet-stream',
          originalBlobId: original ? retain(original) : current?.originalBlobId ?? null,
          parsedBlobId: parsed ? retain(parsed) : current?.parsedBlobId ?? null })
        if (value.dataUrl && original) images.set(value.dataUrl, id)
        value.managedBackupRef = id
        delete value.dataUrl; delete value.previewDataUrl; delete value.source
      }
    }
    visitTimeline(thread.timeline)
    for (const revision of thread.messageEditRevisions ?? []) visitTimeline(revision.timeline)
    const visitMessages = (messages: Thread['apiMessages']) => {
      for (const message of messages ?? []) if (Array.isArray(message.content)) for (const block of message.content) {
        if (block.type !== 'image_url' || !block.image_url.url.startsWith('data:')) continue
        let id = images.get(block.image_url.url)
        const bytes = imageBytes(block.image_url.url)
        if (!id && bytes) {
          id = createHash('sha256').update(`${thread.id}\0${block.image_url.url}`).digest('hex')
          descriptors.set(id, { id, name: 'Saved image.png', mediaType: /^data:([^;]+)/u.exec(block.image_url.url)?.[1] ?? 'image/png', originalBlobId: retain(bytes), parsedBlobId: null })
        }
        block.image_url.url = id ? `backup-image:${id}` : 'backup-image:unavailable'
      }
    }
    visitMessages(thread.apiMessages)
    for (const revision of thread.messageEditRevisions ?? []) visitMessages(revision.apiMessages)
  }
  return { attachments: [...descriptors.values()], blobs }
}

export function hydrateManagedAttachments(threads: Thread[], managed: ManagedAttachmentSnapshot): void {
  const descriptors = new Map(managed.attachments.map((item) => [item.id, item]))
  for (const thread of threads) {
    const visitTimeline = (timeline: Thread['timeline']) => {
      for (const item of timeline ?? []) for (const attachment of item.attachments ?? []) {
        const value = attachment as Attachment & { managedBackupRef?: string }, descriptor = descriptors.get(value.managedBackupRef ?? '')
        if (!descriptor) continue
        const original = descriptor.originalBlobId ? managed.blobs.get(descriptor.originalBlobId) : null
        const parsed = descriptor.parsedBlobId ? managed.blobs.get(descriptor.parsedBlobId) : null
        if (value.kind === 'image' && original) value.dataUrl = `data:${descriptor.mediaType};base64,${Buffer.from(original).toString('base64')}`
        if (parsed) value.text = new TextDecoder('utf-8', { fatal: true }).decode(parsed)
        if ((value.kind === 'image' && !original) || (value.kind === 'document' && !parsed)) {
          value.status = 'needs-reselection'; value.error = 'The backup has no usable managed file content; select the original again.'
        }
      }
    }
    visitTimeline(thread.timeline)
    for (const revision of thread.messageEditRevisions ?? []) visitTimeline(revision.timeline)
    const visitMessages = (messages: Thread['apiMessages']) => {
      for (const message of messages ?? []) if (Array.isArray(message.content)) {
        message.content = message.content.map((block) => {
          if (block.type !== 'image_url' || !block.image_url.url.startsWith('backup-image:')) return block
          const descriptor = descriptors.get(block.image_url.url.slice('backup-image:'.length))
          const original = descriptor?.originalBlobId ? managed.blobs.get(descriptor.originalBlobId) : null
          return original && descriptor ? { ...block, image_url: { ...block.image_url, url: `data:${descriptor.mediaType};base64,${Buffer.from(original).toString('base64')}` } }
            : { type: 'text' as const, text: '[Saved image is unavailable; select the original again.]' }
        })
      }
    }
    visitMessages(thread.apiMessages)
    for (const revision of thread.messageEditRevisions ?? []) visitMessages(revision.apiMessages)
  }
}

export const serializeBackupRecord = (value: unknown): BackupRecord => JSON.parse(JSON.stringify(value)) as BackupRecord
