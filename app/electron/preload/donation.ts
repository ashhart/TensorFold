import type { DonationApi, DonationHostInput } from '../../shared/donation'

export function donationPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): DonationApi {
  return {
    donationList: () => invoke('donation:list'),
    donationImport: (input) => invoke('donation:import', input),
    donationRemove: (input) => invoke('donation:remove', input),
    donationProbe: (input) => invoke('donation:probe', input),
    donationHostStart: (input: DonationHostInput) => invoke('donation:hostStart', input),
    donationHostStop: () => invoke('donation:hostStop'),
    donationHostStatus: () => invoke('donation:hostStatus'),
  }
}
