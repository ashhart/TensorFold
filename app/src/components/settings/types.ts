import type { Dispatch, MutableRefObject, SetStateAction } from 'react'
import type { Settings as TSettings } from '../../lib/types'

// Shared dependency bag passed from Settings.tsx into each extracted settings
// hook. Settings.tsx owns the draft + the three commit primitives; hooks hang
// feature-specific state and handlers off them and return an object that is
// spread into sectionCtx.
export interface SettingsHookDeps {
  draft: TSettings
  // editDraft is `typeof setDraft` from Settings.tsx — marks the draft dirty so
  // an external save can't clobber in-progress edits, then forwards to setDraft.
  editDraft: Dispatch<SetStateAction<TSettings>>
  // Raw setters are reserved for backend-owned status reconciliation.
  setDraft: Dispatch<SetStateAction<TSettings>>
  commit: (next: TSettings) => void
  patch: (partial: Partial<TSettings>) => void
  draftRef?: MutableRefObject<TSettings>
  // Provider setup waits for the same owned draft transaction before reporting success.
  save?: (next: TSettings) => void | Promise<void>
}
