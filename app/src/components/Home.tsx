import { Composer } from './Composer'
import { RecentWorkspaceTasks, WorkspaceWelcome } from './workspace/WorkspaceWelcome'

// Keep the draft composer mounted while the welcome surface provides workspace navigation.
export function Home() {
  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-5 py-8 sm:px-6 sm:py-12 [@media(max-height:650px)]:py-5">
        <WorkspaceWelcome />
        <div className="mt-6 [@media(max-height:650px)]:mt-4">
          <Composer mode="home" />
        </div>
        <RecentWorkspaceTasks />
      </div>
    </div>
  )
}
