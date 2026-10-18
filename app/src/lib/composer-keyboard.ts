// WebKit can report keyCode 229 on the Enter that commits an IME candidate
// even after isComposing becomes false; neither event is a submit/navigation.
export function isTextComposition(event: { keyCode?: number; nativeEvent: { isComposing?: boolean; keyCode?: number } }): boolean {
  return event.nativeEvent.isComposing === true || event.keyCode === 229 || event.nativeEvent.keyCode === 229
}

/** Does not intercept editing, composition, repeated keys or Alt/Shift newline gestures. */
export function isComposerSendKey(event: { key: string; repeat?: boolean; shiftKey?: boolean; altKey?: boolean; metaKey?: boolean; ctrlKey?: boolean; keyCode?: number; nativeEvent: { isComposing?: boolean; keyCode?: number } }, preference?: 'enter' | 'modified-enter'): boolean {
  if (isTextComposition(event) || event.repeat || event.key !== 'Enter' || event.shiftKey || event.altKey) return false
  const modified = event.metaKey === true || event.ctrlKey === true
  return preference === 'modified-enter' ? modified : !modified
}
