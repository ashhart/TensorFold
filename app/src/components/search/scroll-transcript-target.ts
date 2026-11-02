/** scrollIntoView also scrolls overflow-hidden ancestors, displacing a docked composer. */
export function scrollTranscriptTarget(container: HTMLElement, target: HTMLElement, behavior: ScrollBehavior = 'auto'): void {
  const viewport = container.getBoundingClientRect(), item = target.getBoundingClientRect()
  const desired = container.scrollTop + item.top - viewport.top - Math.max(0, (container.clientHeight - item.height) / 2)
  const maximum = Math.max(0, container.scrollHeight - container.clientHeight)
  const top = Math.max(0, Math.min(maximum, desired))
  if (behavior === 'smooth') container.scrollTo({ top, behavior })
  else container.scrollTop = top
}
