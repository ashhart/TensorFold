import { Composer } from './Composer'
import { BrandMark } from './BrandMark'

// Home is an empty thread with no hero heading. The quiet logo floats in the empty
// transcript space and the composer docks at the bottom, where it stays once
// messages start flowing.
export function Home() {
  return (
    <main className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1 select-none items-center justify-center">
        <BrandMark size={150} className="fade-in opacity-90" />
      </div>
      <div className="relative shrink-0 px-6 pb-5">
        <div className="mx-auto w-full max-w-3xl">
          <Composer mode="home" />
        </div>
      </div>
    </main>
  )
}
