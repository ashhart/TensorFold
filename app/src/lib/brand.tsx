import type { CSSProperties, ReactNode } from 'react'
import clsx from 'clsx'
import type { ConnectorId, Provider } from './types'

export type BrandId =
  | 'openai'
  | 'codex'
  | 'claude'
  | 'anthropic'
  | 'openrouter'
  | 'gateway'
  | 'opencode'
  | 'omlx'
  | 'ollama'
  | 'lmstudio'
  | 'telegram'
  | 'whatsapp'
  | 'email'
  | 'outlook'
  | 'slack'
  | 'discord'
  | 'mattermost'
  | 'matrix'
  | 'googlechat'
  | 'calendar'
  | 'docs'
  | 'browser'
  | 'imessage'
  | 'dynamics'
  | 'github'
  | 'mcp'
  | 'controller'
  | 'groq'
  | 'together'
  | 'moonshot'
  | 'minimax'
  | 'deepseek'
  | 'xai'
  | 'zai'
  | 'local'
  | 'custom'

interface BrandMeta {
  label: string
  bg: string
  fg: string
}

const META: Record<BrandId, BrandMeta> = {
  openai: { label: 'OpenAI', bg: '#0f1115', fg: '#ffffff' },
  codex: { label: 'Codex', bg: '#10131a', fg: '#ffffff' },
  claude: { label: 'Claude', bg: '#d97745', fg: '#fff7ed' },
  anthropic: { label: 'Anthropic', bg: '#191716', fg: '#f7f2ea' },
  openrouter: { label: 'OpenRouter', bg: '#111827', fg: '#6ee7b7' },
  gateway: { label: 'AI Gateway', bg: '#111111', fg: '#fafafa' },
  opencode: { label: 'OpenCode', bg: '#f97316', fg: '#fff7ed' },
  omlx: { label: 'oMLX', bg: '#4638c2', fg: '#ffffff' },
  ollama: { label: 'Ollama', bg: '#111111', fg: '#ffffff' },
  lmstudio: { label: 'LM Studio', bg: '#1d4ed8', fg: '#eff6ff' },
  telegram: { label: 'Telegram', bg: '#229ed9', fg: '#ffffff' },
  whatsapp: { label: 'WhatsApp', bg: '#25d366', fg: '#ffffff' },
  email: { label: 'Email', bg: '#2563eb', fg: '#eff6ff' },
  outlook: { label: 'Outlook', bg: '#0078d4', fg: '#ffffff' },
  slack: { label: 'Slack', bg: '#ffffff', fg: '#1d1c1d' },
  discord: { label: 'Discord', bg: '#5865f2', fg: '#ffffff' },
  mattermost: { label: 'Mattermost', bg: '#1e325c', fg: '#ffffff' },
  matrix: { label: 'Matrix', bg: '#0dbd8b', fg: '#0a0a0a' },
  googlechat: { label: 'Google Chat', bg: '#00897b', fg: '#ffffff' },
  calendar: { label: 'Calendar', bg: '#f59e0b', fg: '#fff7ed' },
  docs: { label: 'Documents', bg: '#16a34a', fg: '#f0fdf4' },
  browser: { label: 'Browser', bg: '#0f172a', fg: '#bae6fd' },
  imessage: { label: 'iMessage', bg: '#34c759', fg: '#ffffff' },
  dynamics: { label: 'Dynamics 365', bg: '#0f6cbd', fg: '#ffffff' },
  github: { label: 'GitHub', bg: '#181717', fg: '#ffffff' },
  mcp: { label: 'MCP', bg: '#4f46e5', fg: '#ffffff' },
  controller: { label: 'Controller', bg: '#0f172a', fg: '#93c5fd' },
  groq: { label: 'Groq', bg: '#f55036', fg: '#fff7ed' },
  together: { label: 'Together', bg: '#7c3aed', fg: '#ffffff' },
  moonshot: { label: 'Moonshot', bg: '#111827', fg: '#dbeafe' },
  minimax: { label: 'MiniMax', bg: '#0ea5e9', fg: '#ffffff' },
  deepseek: { label: 'DeepSeek', bg: '#2563eb', fg: '#eff6ff' },
  xai: { label: 'xAI', bg: '#111111', fg: '#ffffff' },
  zai: { label: 'z.ai', bg: '#111827', fg: '#a7f3d0' },
  local: { label: 'Local model', bg: '#0f766e', fg: '#ecfeff' },
  custom: { label: 'Custom provider', bg: '#3f3f46', fg: '#fafafa' },
}

export function providerBrand(provider: Partial<Provider>): BrandId {
  const name = `${provider.name ?? ''} ${provider.baseUrl ?? ''} ${provider.source ?? ''}`.toLowerCase()
  if (provider.transport === 'codex-cli' || /\bcodex\b/.test(name)) return 'codex'
  if (provider.transport === 'claude-cli' || /\bclaude\b/.test(name)) return 'claude'
  if (/anthropic/.test(name)) return 'anthropic'
  if (provider.source === 'openrouter' || /openrouter/.test(name)) return 'openrouter'
  if (provider.source === 'ai-gateway' || /ai-gateway\.vercel\.sh|ai gateway|vercel gateway/.test(name)) return 'gateway'
  if (provider.source === 'omlx' || /\bomlx\b/.test(name)) return 'omlx'
  if (provider.source === 'ollama' || /ollama/.test(name)) return 'ollama'
  if (provider.source === 'lmstudio' || /lm\s*studio/.test(name)) return 'lmstudio'
  if (/groq/.test(name)) return 'groq'
  if (/together/.test(name)) return 'together'
  if (/moonshot/.test(name)) return 'moonshot'
  if (/minimax/.test(name)) return 'minimax'
  if (/deepseek/.test(name)) return 'deepseek'
  if (/\bxai\b|grok|api\.x\.ai/.test(name)) return 'xai'
  if (/\bz\.ai\b|glm/.test(name)) return 'zai'
  if (provider.source === 'openai' || /api\.openai\.com|(^|\s)openai(\s|$)/.test(name)) return 'openai'
  if (provider.kind === 'local') return 'local'
  return 'custom'
}

export function connectorBrand(id: ConnectorId | string): BrandId {
  if (id === 'telegram') return 'telegram'
  if (id === 'whatsapp') return 'whatsapp'
  if (id === 'email') return 'email'
  if (id === 'slack') return 'slack'
  if (id === 'discord') return 'discord'
  if (id === 'mattermost') return 'mattermost'
  if (id === 'matrix') return 'matrix'
  if (id === 'googlechat') return 'googlechat'
  if (id === 'calendar') return 'calendar'
  if (id === 'docs') return 'docs'
  if (id === 'browser') return 'browser'
  if (id === 'outlook') return 'outlook'
  if (id === 'imessage') return 'imessage'
  if (id === 'dynamics') return 'dynamics'
  if (id === 'github') return 'github'
  if (id === 'controller') return 'controller'
  return 'custom'
}

export function brandLabel(brand: BrandId): string {
  return META[brand].label
}

export function BrandIcon({ brand, size = 28, className, title }: { brand: BrandId; size?: number; className?: string; title?: string }) {
  const meta = META[brand]
  const style = {
    width: size,
    height: size,
    color: meta.fg,
    background: meta.bg,
  } satisfies CSSProperties
  return (
    <span
      className={clsx('inline-flex shrink-0 items-center justify-center overflow-hidden rounded-lg border border-black/10 shadow-[inset_0_1px_rgba(255,255,255,.16)]', className)}
      style={style}
      title={title ?? meta.label}
      aria-label={title ?? meta.label}
    >
      <svg width={Math.max(14, Math.round(size * 0.66))} height={Math.max(14, Math.round(size * 0.66))} viewBox="0 0 24 24" aria-hidden="true">
        {MARKS[brand]}
      </svg>
    </span>
  )
}

const Letter = ({ children, size = 8.8 }: { children: ReactNode; size?: number }) => (
  <text x="12" y="15" textAnchor="middle" fontSize={size} fontWeight="800" fontFamily="Inter, ui-sans-serif, system-ui" fill="currentColor">{children}</text>
)

const MARKS: Record<BrandId, ReactNode> = {
  telegram: <path fill="currentColor" d="M20.5 4.4 3.8 10.9c-1.1.4-1.1 1.1-.2 1.4l4.2 1.3 1.6 5c.2.6.4.8.8.8s.6-.2.9-.5l2.2-2.1 4.5 3.3c.8.4 1.3.2 1.5-.8l2.7-12.9c.3-1.2-.4-1.7-1.5-1.2Zm-3.4 3.4-7.6 6.9-.3 2.4-1-3.3 8.9-6Z" />,
  whatsapp: (
    <>
      <path fill="currentColor" d="M12 3.2a8.2 8.2 0 0 0-7.1 12.3L4 20.8l5.4-1.4A8.2 8.2 0 1 0 12 3.2Zm0 14.6c-1.3 0-2.5-.4-3.5-1l-.2-.2-2.3.6.6-2.2-.2-.3A6.3 6.3 0 1 1 12 17.8Z" opacity=".95" />
      <path fill={META.whatsapp.bg} d="M8.6 8.1c.2-.5.4-.5.7-.5h.5c.2 0 .4 0 .6.4l.7 1.7c.1.3.1.5 0 .6l-.4.5c-.1.1-.3.3-.1.6.2.4.8 1.3 1.7 2 .8.7 1.5.9 1.8 1 .3.1.5 0 .7-.2l.7-.8c.2-.2.4-.2.6-.1l1.7.8c.3.2.5.3.5.5 0 .7-.6 1.7-1.2 2-.3.2-.8.3-1.5.2-2.5-.4-5.2-2.7-6.5-4.5-.9-1.2-1.4-2.4-1.2-3.2.1-.4.4-.8.7-1Z" />
    </>
  ),
  claude: (
    <>
      <path fill="currentColor" d="M12 3.2 13.7 9l5.8-1.7-4.1 4.7 4.1 4.7-5.8-1.7L12 20.8 10.3 15l-5.8 1.7 4.1-4.7-4.1-4.7L10.3 9z" />
      <circle cx="12" cy="12" r="2" fill={META.claude.bg} />
    </>
  ),
  codex: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="3" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M9 10 6.8 12 9 14M15 10l2.2 2L15 14M11 16l2-8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </>
  ),
  openai: (
    <>
      <path d="M12 3.6c1.4 0 2.7.7 3.5 1.8 1.5-.1 3 .7 3.8 2.1.7 1.3.7 2.8 0 4.1.7 1.3.7 2.9 0 4.2-.8 1.4-2.3 2.1-3.8 2.1A4.4 4.4 0 0 1 12 20.4c-1.4 0-2.7-.7-3.5-1.8-1.5.1-3-.7-3.8-2.1-.7-1.3-.7-2.8 0-4.1-.7-1.3-.7-2.9 0-4.2.8-1.4 2.3-2.1 3.8-2.1A4.4 4.4 0 0 1 12 3.6Z" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M8.5 6.1 15.3 10v7.9M18.8 11.5 12 15.4 5.2 11.5M15.5 6.1 8.7 10v7.9" fill="none" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" />
    </>
  ),
  anthropic: <Letter size={9.5}>A</Letter>,
  openrouter: (
    <>
      <path d="M4 8h9.8c2.5 0 4.2 1.5 4.2 4s-1.7 4-4.2 4H4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M14.5 5.7 18 8l-3.5 2.3M9.5 13.7 6 16l3.5 2.3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </>
  ),
  gateway: (
    <>
      <path d="M4 12h4.5M15.5 12H20M12 4v4.5M12 15.5V20" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
      <circle cx="12" cy="12" r="3.7" fill="none" stroke="currentColor" strokeWidth="1.9" />
      <path d="M9.4 9.4 6.8 6.8M14.6 9.4l2.6-2.6M14.6 14.6l2.6 2.6M9.4 14.6l-2.6 2.6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </>
  ),
  opencode: <Letter size={7.4}>OC</Letter>,
  omlx: <Letter size={7.2}>MLX</Letter>,
  ollama: <Letter size={11}>O</Letter>,
  lmstudio: <Letter size={8.4}>LM</Letter>,
  email: <path d="M4 7h16v10H4zM4.8 7.8 12 13l7.2-5.2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />,
  outlook: (
    <>
      <rect x="5" y="6" width="9" height="12" rx="1.5" fill="currentColor" opacity=".9" />
      <path d="M13 8h6v8h-6zM13 9l3 2.6L19 9" fill="none" stroke="currentColor" strokeWidth="1.5" opacity=".65" />
      <text x="9.5" y="14.8" textAnchor="middle" fontSize="7" fontWeight="800" fill={META.outlook.bg}>O</text>
    </>
  ),
  slack: (
    <>
      <path fill="#36C5F0" d="M8.2 4a2 2 0 1 1 4 0v4.2h-4z" />
      <path fill="#2EB67D" d="M20 8.2a2 2 0 1 1 0 4h-4.2v-4z" />
      <path fill="#ECB22E" d="M15.8 20a2 2 0 1 1-4 0v-4.2h4z" />
      <path fill="#E01E5A" d="M4 15.8a2 2 0 1 1 0-4h4.2v4z" />
      <path fill="#36C5F0" d="M8.2 8.2H6a2 2 0 0 1 2.2-2.2z" />
      <path fill="#2EB67D" d="M15.8 8.2V6A2 2 0 0 1 18 8.2z" />
      <path fill="#ECB22E" d="M15.8 15.8H18a2 2 0 0 1-2.2 2.2z" />
      <path fill="#E01E5A" d="M8.2 15.8V18A2 2 0 0 1 6 15.8z" />
    </>
  ),
  discord: <path fill="currentColor" d="M19.3 5.3A17 17 0 0 0 15 4l-.3.5a13 13 0 0 1 3.6 1.2A14 14 0 0 0 6 6.7 13 13 0 0 1 9.5 5.5L9 4a17 17 0 0 0-4.3 1.3C2 9.4 1.3 13.4 1.6 17.3a17 17 0 0 0 5.2 2.6l.6-1a11 11 0 0 1-2-1l.5-.4a12 12 0 0 0 10.3 0l.5.4a11 11 0 0 1-2 1l.6 1a17 17 0 0 0 5.2-2.6c.4-4.5-.6-8.5-3.2-12zM8.3 14.8c-.8 0-1.5-.8-1.5-1.7s.7-1.7 1.5-1.7 1.5.8 1.5 1.7-.7 1.7-1.5 1.7zm7.4 0c-.8 0-1.5-.8-1.5-1.7s.7-1.7 1.5-1.7 1.5.8 1.5 1.7-.7 1.7-1.5 1.7z" />,
  mattermost: <Letter size={7.2}>MM</Letter>,
  matrix: <Letter size={10}>M</Letter>,
  googlechat: <path d="M5 4h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H9l-4 3v-3a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />,
  calendar: <path d="M6 5h12a2 2 0 0 1 2 2v11H4V7a2 2 0 0 1 2-2Zm-2 5h16M8 3v4M16 3v4M8 13h3v3H8z" fill="none" stroke="currentColor" strokeWidth="1.8" />,
  docs: <path d="M7 3h7l4 4v14H7zM14 3v5h4M10 12h5M10 16h5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />,
  browser: <path d="M4 6h16v12H4zM4 9.5h16M7 7.8h.01M10 7.8h.01M8 13h8M8 15.5h5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />,
  imessage: <path d="M12 5c4.4 0 8 2.8 8 6.3s-3.6 6.3-8 6.3c-.9 0-1.8-.1-2.6-.4L5 19l1.4-3.1C4.9 14.8 4 13.1 4 11.3 4 7.8 7.6 5 12 5Z" fill="currentColor" />,
  dynamics: <path d="M6 4 18 8v8L6 20V4Zm3.2 4.3v7.4l5.5-2v-3.4z" fill="currentColor" />,
  github: (
    <>
      <circle cx="12" cy="11.5" r="7" fill="currentColor" />
      <path d="M9.3 17.8v-2.2c-2 .5-2.7-.6-3-1.3M14.7 17.8v-2.2c0-.8-.3-1.2-.7-1.5 2.3-.3 4.1-1.1 4.1-4.1 0-.9-.3-1.6-.8-2.2.1-.3.3-1.2-.1-2.1 0 0-.7-.2-2.3.8a7.8 7.8 0 0 0-5.8 0c-1.6-1-2.3-.8-2.3-.8-.4.9-.2 1.8-.1 2.1A3.2 3.2 0 0 0 6 10c0 3 1.8 3.8 4.1 4.1-.3.2-.5.6-.6 1" fill="none" stroke={META.github.bg} strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
    </>
  ),
  mcp: <path d="M12 3 4.5 7.4v9.2L12 21l7.5-4.4V7.4L12 3Zm0 0v18M4.8 7.5 12 12l7.2-4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />,
  controller: <path d="M4 5.5h16a1.8 1.8 0 0 1 1.8 1.8v9.4a1.8 1.8 0 0 1-1.8 1.8H4a1.8 1.8 0 0 1-1.8-1.8V7.3A1.8 1.8 0 0 1 4 5.5Zm4 4h8M8 12h5M8 14.5h8M18.5 8.5v7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />,
  groq: <Letter size={9}>G</Letter>,
  together: <Letter size={8.2}>T</Letter>,
  moonshot: <path d="M17.8 15.6A7 7 0 1 1 8.4 6.2 5.4 5.4 0 0 0 17.8 15.6Z" fill="currentColor" />,
  minimax: <Letter size={7.4}>MX</Letter>,
  deepseek: <Letter size={7.2}>DS</Letter>,
  xai: <Letter size={8.6}>xAI</Letter>,
  zai: <Letter size={9}>Z</Letter>,
  local: <path d="M8 8h8v8H8zM10 3v3M14 3v3M10 18v3M14 18v3M3 10h3M3 14h3M18 10h3M18 14h3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
  custom: <path d="M12 3 4 7.5v9L12 21l8-4.5v-9zM12 3v18M4.4 7.7 12 12l7.6-4.3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />,
}
