import type { DonationApi } from '../../../shared/donation'

/** A browser preview has no profile, no donor process and no sockets to check. */
export function createMockDonationApi(): DonationApi {
  const unavailable = async (): Promise<never> => { throw new Error('Lending and borrowing compute requires the desktop app; no machine was contacted.') }
  return {
    donationList: async () => ({ cards: [], errors: [] }),
    donationImport: unavailable,
    donationRemove: unavailable,
    donationProbe: unavailable,
    donationHostStart: unavailable,
    donationHostStop: async () => ({ state: 'idle' }),
    donationHostStatus: async () => ({ state: 'idle' }),
  }
}
