import { useComposerState, type ComposerProps } from './composer/useComposerState'
import { useComposerWorkspace } from './composer/useComposerWorkspace'
import { useComposerSubmit } from './composer/useComposerSubmit'
import { ComposerView } from './composer/ComposerView'

export function Composer(props: ComposerProps) {
  const state = useComposerState(props)
  const workspace = useComposerWorkspace(state)
  const submission = useComposerSubmit({ ...state, ...workspace })
  return <ComposerView {...state} {...workspace} {...submission} />
}
