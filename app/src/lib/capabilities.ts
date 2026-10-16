import type { IconName } from './icons'
import type { CapabilityBuiltInId } from './types'

export type BuiltInCapabilityId = CapabilityBuiltInId

export interface BuiltInCapability {
  id: BuiltInCapabilityId
  mention: string
  name: string
  description: string
  icon: IconName
  color: string
  prompt: string
  status: 'available' | 'contract'
}

export const BUILTIN_CAPABILITIES: BuiltInCapability[] = [
  {
    id: 'documents',
    mention: 'documents',
    name: 'Documents',
    description: 'Create and edit document artifacts',
    icon: 'fileText',
    color: '#2563eb',
    status: 'contract',
    prompt: 'Use the Documents capability for Word-style document drafting, editing, summaries, outlines, and artifact generation.',
  },
  {
    id: 'pdf',
    mention: 'pdf',
    name: 'PDF',
    description: 'Read, create, and verify PDF files',
    icon: 'filePdf',
    color: '#dc2626',
    status: 'contract',
    prompt: 'Use the PDF capability for extracting, verifying, summarising, creating, or checking PDF files and PDF-derived evidence.',
  },
  {
    id: 'spreadsheets',
    mention: 'sheets',
    name: 'Spreadsheets',
    description: 'Create and edit spreadsheet files',
    icon: 'table',
    color: '#16a34a',
    status: 'contract',
    prompt: 'Use the Spreadsheets capability for tables, CSV/XLSX-style data work, formulas, analysis, and workbook generation.',
  },
  {
    id: 'presentations',
    mention: 'slides',
    name: 'Presentations',
    description: 'Create and edit presentations',
    icon: 'presentation',
    color: '#d97706',
    status: 'contract',
    prompt: 'Use the Presentations capability for slide outlines, speaker notes, deck structure, and presentation artifact generation.',
  },
  {
    id: 'browser',
    mention: 'browser',
    name: 'Browser',
    description: 'Automate browser QA with snapshots and evidence',
    icon: 'browser',
    color: '#0ea5e9',
    status: 'available',
    prompt: 'Use the Browser capability for approval-gated navigation, clicks, typing, form fill, tabs, screenshots, console/network inspection, and accessibility snapshots.',
  },
  {
    id: 'controller',
    mention: 'controller',
    name: 'Controller',
    description: 'Attach context and approve scoped Mac app actions',
    icon: 'monitor',
    color: '#8b5cf6',
    status: 'available',
    prompt: 'Use the Controller capability for explicit current-window context and approval-gated Mac app actions. Keep every action scoped to the requested app and visible approval.',
  },
  {
    id: 'computer',
    mention: 'computer',
    name: 'Computer use',
    description: 'Operate Mac apps with approval per app',
    icon: 'monitor',
    color: '#5b4bd8',
    status: 'available',
    prompt: 'Computer use is on for this task. Operate Mac apps with the computer_use tool: open_app, then observe to list the controls, then act on them by element id, naming the app on every call. Observe again after anything that changes the window. Ask the user before anything irreversible such as sending, buying or deleting, and never type passwords.',
  },
]

export function capabilityToken(capability: BuiltInCapability): string {
  return `@${capability.mention}`
}

export function capabilityInstruction(capability: BuiltInCapability): string {
  return [
    `<vontra_capability name="${capability.name}" id="${capability.id}">`,
    capability.prompt,
    '</vontra_capability>',
  ].join('\n')
}
