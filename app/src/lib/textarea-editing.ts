/** Use Chromium's editor for replacements so native selection and Undo/Redo are preserved. */
export function insertTextareaText(textarea: HTMLTextAreaElement, insert: string, start = textarea.selectionStart, end = textarea.selectionEnd): string {
  textarea.focus()
  textarea.setSelectionRange(start, end)
  const before = textarea.value
  const next = before.slice(0, start) + insert + before.slice(end)
  const inserted = document.execCommand('insertText', false, insert)
  if (!inserted || textarea.value !== next) {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
    setter?.call(textarea, next)
    textarea.setSelectionRange(start + insert.length, start + insert.length)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  }
  return next
}
