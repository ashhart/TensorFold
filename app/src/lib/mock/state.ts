import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'

export const uid = () => (crypto.randomUUID ? crypto.randomUUID() : 'id-' + Math.random().toString(36).slice(2))
export const ago = (min: number) => Date.now() - min * 60_000
export function startOfMockDay(hour = 0, minute = 0): number {
  const date = new Date()
  date.setHours(hour, minute, 0, 0)
  return date.getTime()
}

export const mockState = {
  settings: null as unknown as Settings,
  sampleDiff: [] as DiffLine[],
  sampleVisualization: undefined as TimelineItem['visualization'],
  threads: [] as Thread[],
  // First-class artifacts (VST-011 demo).
  mockArtifacts: [] as import('../types').Artifact[],
  // Repo-authored project skills (VST-010 demo).
  mockProjectSkills: undefined as import('../types').ProjectSkillView[] | undefined,
  mockSuggestions: undefined as import('../types').SkillSuggestion[] | undefined,
  // Paired device sessions (VST-035 demo).
  deviceSessions: undefined as Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }> | undefined,
  listener: null as ((e: AgentEvent) => void) | null,
  projects: [] as Project[],
  mockRooms: [] as Room[],
  roomListener: null as ((e: RoomStreamEvent) => void) | null,
  mockTickets: [] as Ticket[],
  ticketListener: null as ((t: Ticket) => void) | null,
  mockJobs: [] as Job[],
  jobListener: null as ((j: Job) => void) | null,
  mockAutomations: [] as Automation[],
  autoListener: null as ((a: Automation) => void) | null,
  updateListener: null as ((s: AppUpdateStatus) => void) | null,
  mockUpdateStatus: null as unknown as AppUpdateStatus,
  mockConnectorEvents: [] as ConnectorEvent[],
  mockEmailMessages: [] as EmailReadResult[],
  mockSlackMessages: [] as SlackReadResult[],
  mockCalendarEvents: [] as CalendarReadResult[],
  mockDocuments: [] as DocumentReadResult[],
  mockBrowserItems: [] as BrowserReadResult[],
  mockMissions: [] as Mission[],
  mockSkills: [] as Skill[],
  mockProjectMemory: [] as ProjectMemoryEntry[],
  mockProjectMemoryCandidates: [] as ProjectMemoryCandidate[],
  mockWebhooks: [] as WebhookEndpoint[],
  mockIncidents: [] as Incident[],
  mockCommands: [] as Command[],
  mockMcpServers: [] as McpServerConfig[],
  mockMcpStopped: new Set<string>(),
  mockGitHygieneSanitized: false,
}
