const workspaceMutationTails = new Map<string, Promise<void>>()
const projectUpdateFieldVersions = new Map<string, number>()
const projectUpdateCommittedValues = new Map<string, unknown>()
const projectUpdateLatestVersions = new Map<string, number>()
let projectUpdateSequence = 0

export function enqueueWorkspaceMutation<T>(keys: readonly string[], task: () => Promise<T>): Promise<T> {
  const orderedKeys = [...new Set(keys)].sort()
  const dependencies = [...new Set(orderedKeys.map((key) => workspaceMutationTails.get(key)).filter((tail): tail is Promise<void> => !!tail))]
  const result = dependencies.length
    ? Promise.all(dependencies.map((tail) => tail.catch(() => undefined))).then(task)
    : task()
  const tail = result.then(() => undefined, () => undefined)
  for (const key of orderedKeys) workspaceMutationTails.set(key, tail)
  void tail.finally(() => {
    for (const key of orderedKeys) if (workspaceMutationTails.get(key) === tail) workspaceMutationTails.delete(key)
  })
  return result
}

export function enqueueProjectUpdate<T>(projectId: string, task: () => Promise<T>): Promise<T> {
  return enqueueWorkspaceMutation([`project:${projectId}`], task)
}

const automationUpdateTails = new Map<string, Promise<void>>()

export function enqueueAutomationUpdate<T>(automationId: string, task: () => Promise<T>): Promise<T> {
  const previous = automationUpdateTails.get(automationId) ?? Promise.resolve()
  const result = previous.catch(() => undefined).then(task)
  const tail = result.then(() => undefined, () => undefined)
  automationUpdateTails.set(automationId, tail)
  void tail.finally(() => { if (automationUpdateTails.get(automationId) === tail) automationUpdateTails.delete(automationId) })
  return result
}

export function enqueueRoomMutation<T>(roomId: string, task: () => Promise<T>): Promise<T> {
  return enqueueWorkspaceMutation(['rooms:*', `room:${roomId}`], task)
}

export function enqueueTicketMutation<T>(ticketId: string, task: () => Promise<T>): Promise<T> {
  return enqueueWorkspaceMutation(['tickets:*', `ticket:${ticketId}`], task)
}

export const projectUpdateState = {
  get sequence() { return projectUpdateSequence },
  bumpSequence() { return ++projectUpdateSequence },
  fieldVersions: projectUpdateFieldVersions,
  committedValues: projectUpdateCommittedValues,
  latestVersions: projectUpdateLatestVersions,
}
