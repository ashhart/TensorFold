// WebKit can report keyCode 229 on the Enter that commits an IME candidate
// even after isComposing becomes false; neither event is a submit/navigation.
export function isTextComposition(event: { keyCode?: number; nativeEvent: { isComposing?: boolean; keyCode?: number } }): boolean {
  return event.nativeEvent.isComposing === true || event.keyCode === 229 || event.nativeEvent.keyCode === 229
}
