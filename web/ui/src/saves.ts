// Serialize saves per conversation and server revision; wait for all replies before reopening a conversation.

export type Saved = { id: string; rev?: number }
export type SaveApi = {
  create: (body: Record<string, unknown>) => Promise<Saved>
  save: (id: string, body: Record<string, unknown>) => Promise<Saved>
}
// 'conflict': the server holds a newer revision; this page stops saving that conversation until it is opened again
export type SaveOutcome = 'saved' | 'failed' | 'conflict'

type Queue = { id: string | null; rev?: number; conflict: boolean; chain: Promise<void> }

export class Saves {
  private byGen = new Map<number, Queue>()
  private byId = new Map<string, Queue>()
  private api: SaveApi
  private onCreated: (gen: number, id: string) => void
  private onDone: (outcome: SaveOutcome, id: string | null) => void
  constructor(api: SaveApi, onCreated: (gen: number, id: string) => void, onDone: (outcome: SaveOutcome, id: string | null) => void) {
    this.api = api; this.onCreated = onCreated; this.onDone = onDone
  }

  private queueFor(id: string): Queue {
    let q = this.byId.get(id)
    if (!q) { q = { id, conflict: false, chain: Promise.resolve() }; this.byId.set(id, q) }
    return q
  }


  read(id: string, rev: number | undefined): void {
    const q = this.queueFor(id)
    if (q.conflict) {
      // Drop queued writes after a revision conflict and start a new queue from the newly read revision.
      this.byId.set(id, { id, rev, conflict: false, chain: Promise.resolve() })
      return
    }
    q.rev = rev
  }


  settled(): Promise<void> {
    return Promise.all([...this.byId.values(), ...this.byGen.values()].map((q) => q.chain)).then(() => {})
  }

  // `id`: the conversation the save belongs to, as the page knows it (null for one not created yet)
  save(gen: number, id: string | null, body: Record<string, unknown>): Promise<void> {
    let q: Queue
    if (id) q = this.queueFor(id)
    else {
      q = this.byGen.get(gen) ?? { id: null, conflict: false, chain: Promise.resolve() }
      this.byGen.set(gen, q)
    }
    const queue = q
    queue.chain = queue.chain.then(async () => {
      if (queue.conflict) return                       // said once; the page opens it again to go on
      try {
        if (queue.id) {
          const r = await this.api.save(queue.id, queue.rev != null ? { ...body, rev: queue.rev } : body)
          queue.rev = r.rev
        } else {
          const r = await this.api.create(body)
          queue.id = r.id; queue.rev = r.rev; this.byId.set(r.id, queue); this.onCreated(gen, r.id)
        }
        this.onDone('saved', queue.id)
      } catch (e) {
        if ((e as { status?: number }).status === 409) { queue.conflict = true; this.onDone('conflict', queue.id) }
        else this.onDone('failed', queue.id)          // the next save tries again (and creates, if none was made)
      }
    })
    for (const g of this.byGen.keys()) if (g < gen - 8) this.byGen.delete(g)
    return queue.chain
  }
}
