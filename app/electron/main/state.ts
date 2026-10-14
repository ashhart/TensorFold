import type { BrowserWindow } from 'electron'
import type { ModelRouteState } from '../agent/model-router'
import type { Artifact,
  AppUpdateStatus,
  Automation,
  BotControlSession,
  Command,
  ConnectorEvent,
  Incident,
  Job,
  McpServerConfig,
  Mission,
  ModelRouterHistoryEntry,
  Project,
  ProjectMemoryCandidate,
  ProjectMemoryEntry,
  Room,
  Settings,
  Skill,
  SkillSuggestion,
  SubAgent,
  Thread,
  Ticket,
  WebhookEndpoint,
  CliSession,
} from '../agent/types'
import type { ContactDirectory } from '../agent/contact-directory'
import { emptyContactDirectory } from '../agent/contact-directory'

// Central mutable state for the Electron main process.
//
// Historically these lived as ~two dozen module-level `let` bindings inside the
// 7k-line main.ts, which is why behaviour-bearing code could not move out of that
// file: every function closed over those bindings. They now live on a single
// shared `store` object so domain modules under electron/main/** can read and
// mutate the same source of truth via `store.x` / `store.x = ...`.
//
// Reassign collections with `store.threads = next` (not a local rebind). The host
// bootstrap (main.ts) seeds `settings`, `updateStatus`, `win`, and the persisted
// collections at startup before anything reads them.
export interface MainState {
  win: BrowserWindow | null
  // First-class artifacts (VST-011).
  artifacts: Artifact[]
  settings: Settings
  threads: Thread[]
  projects: Project[]
  rooms: Room[]
  tickets: Ticket[]
  automations: Automation[]
  connectorEvents: ConnectorEvent[]
  missions: Mission[]
  jobs: Job[]
  subAgents: SubAgent[]
  skills: Skill[]
  commands: Command[]
  suggestions: SkillSuggestion[]
  projectMemory: ProjectMemoryEntry[]
  projectMemoryCandidates: ProjectMemoryCandidate[]
  webhooks: WebhookEndpoint[]
  modelRouteState: ModelRouteState
  modelRouteHistory: ModelRouterHistoryEntry[]
  incidents: Incident[]
  mcpServers: McpServerConfig[]
  botSessions: BotControlSession[]
  // Learned platform-handle -> display-name directory (contact-directory.ts). Populated
  // from what connectors already see (email From names, slack/discord member info,
  // imessage/whatsapp contact fields); resolves raw numbers/ids to names in the feed and
  // a name back to a handle for "reply to <name>". Persisted to ~/.vontra/contacts.json.
  contacts: ContactDirectory
  // The set of project terminals that were LIVE, mirrored to disk (cli-sessions.json) so they
  // survive an app restart and can be resumed (terminal-runtime.ts owns the live↔persisted sync).
  cliSessions: CliSession[]
  updateStatus: AppUpdateStatus
}

export const store: MainState = {
  artifacts: [],
  win: null,
  // Seeded at boot by the host bootstrap before first read.
  settings: undefined as unknown as Settings,
  threads: [],
  projects: [],
  rooms: [],
  tickets: [],
  automations: [],
  connectorEvents: [],
  missions: [],
  jobs: [],
  subAgents: [],
  skills: [],
  commands: [],
  suggestions: [],
  projectMemory: [],
  projectMemoryCandidates: [],
  webhooks: [],
  modelRouteState: { exhausted: [] },
  modelRouteHistory: [],
  incidents: [],
  mcpServers: [],
  botSessions: [],
  contacts: emptyContactDirectory(),
  cliSessions: [],
  // Seeded at boot by the host bootstrap before first read.
  updateStatus: undefined as unknown as AppUpdateStatus,
}
