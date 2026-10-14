import clsx from 'clsx'
import { priorityClass } from './constants'
import type { Priority } from './types'

export function PriorityPill({ priority }: { priority: Priority }) {
  return <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium capitalize', priorityClass[priority])}>{priority}</span>
}
