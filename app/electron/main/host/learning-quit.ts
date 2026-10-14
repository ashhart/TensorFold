interface LearningLifecycle {
  isLearningBusy(): boolean
  prepareToQuit(): Promise<void>
}

// Keep Electron alive until the native learner has finished saving its private checkpoint.
// A repeated Quit must not bypass the fence or run the normal teardown early.
export function createLearningQuitGuard(options: { service(): LearningLifecycle | null; quit(): void; failed(error: unknown): void }) {
  let waiting = false
  return (event: { preventDefault(): void }): boolean => {
    const service = options.service()
    if (!waiting && !service?.isLearningBusy()) return false
    event.preventDefault()
    if (waiting || !service) return true
    waiting = true
    void service.prepareToQuit().then(() => {
      waiting = false
      options.quit()
    }).catch((error) => {
      waiting = false
      options.failed(error)
    })
    return true
  }
}
