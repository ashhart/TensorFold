/**
 * Donation cards on disk. One JSON file per profile under `state/donation.json`, mode 0600:
 * a card carries a bearer token, so it is written private and never echoed back to the renderer
 * with its token intact.
 */
import { promises as fs } from 'node:fs'
import { randomBytes } from 'node:crypto'
import path from 'node:path'
import { parseDonationCard, type DonationCard } from '../../../../shared/donation'

export interface StoredDonationState {
  schema: 1
  cards: DonationCard[]
}

const EMPTY: StoredDonationState = { schema: 1, cards: [] }

export function donationStatePath(root: string): string {
  return path.join(root, 'state', 'donation.json')
}

export async function loadDonationCards(root: string): Promise<{ cards: DonationCard[]; errors: string[] }> {
  let text: string
  try {
    text = await fs.readFile(donationStatePath(root), 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { cards: [], errors: [] }
    throw error
  }
  const errors: string[] = []
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return { cards: [], errors: ['The donation file is not valid JSON.'] }
  }
  const list = typeof raw === 'object' && raw !== null && Array.isArray((raw as { cards?: unknown }).cards) ? (raw as { cards: unknown[] }).cards : []
  const cards: DonationCard[] = []
  for (const entry of list) {
    const parsed = parseDonationCard(entry)
    if ('card' in parsed) cards.push(parsed.card)
    else errors.push(parsed.error)
  }
  return { cards, errors }
}

export async function saveDonationCards(root: string, cards: DonationCard[]): Promise<void> {
  const file = donationStatePath(root)
  await fs.mkdir(path.dirname(file), { recursive: true })
  const state: StoredDonationState = { ...EMPTY, cards }
  const temporary = `${file}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`
  await fs.writeFile(temporary, `${JSON.stringify(state, null, 2)}\n`, { mode: 0o600 })
  await fs.chmod(temporary, 0o600)
  await fs.rename(temporary, file)
}
