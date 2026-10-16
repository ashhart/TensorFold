import React from 'react'

// Crisp 24px line icons, 1.7 stroke.
const P: Record<string, React.ReactNode> = {
  plus: <path d="M12 5v14M5 12h14" />,
  send: <path d="M12 19V6M6 12l6-6 6 6" />,
  arrowDown: <path d="M12 5v14M6 13l6 6 6-6" />,
  mic: <><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8.5 21h7" /></>,
  speaker: <><path d="M4 10v4h4l5 4V6l-5 4z" /><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12" /></>,
  coffee: <><path d="M5 8h11v6.5A4.5 4.5 0 0 1 11.5 19h-2A4.5 4.5 0 0 1 5 14.5z" /><path d="M16 10h2a2.5 2.5 0 0 1 0 5h-2" /><path d="M7 3.5c.8.7.8 1.5 0 2.2M11 3.5c.8.7.8 1.5 0 2.2M15 3.5c.8.7.8 1.5 0 2.2" /><path d="M4 21h14" /></>,
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" />,
  incognito: <><path d="M4 12h16" /><path d="M7.5 12l1.6-5.2A1.8 1.8 0 0 1 10.8 5.5h2.4a1.8 1.8 0 0 1 1.7 1.3L16.5 12" /><circle cx="8.2" cy="16.8" r="2.4" /><circle cx="15.8" cy="16.8" r="2.4" /><path d="M10.6 16.4c.9-.5 1.9-.5 2.8 0" /></>,
  settings: (
    <>
      <circle cx="12" cy="12" r="3.2" />
      <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" />
    </>
  ),
  folder: <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2.2H19.5A1.5 1.5 0 0 1 21 9.7V17.5A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5z" />,
  terminal: <><path d="M5 7l4 5-4 5" /><path d="M12 17h7" /></>,
  code: <><path d="M9 18l-6-6 6-6" /><path d="M15 6l6 6-6 6" /></>,
  file: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>,
  fileText: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /><path d="M8.5 12h7M8.5 15h7M8.5 18H13" /></>,
  filePdf: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /><path d="M8 16v-4h1.6a1.2 1.2 0 1 1 0 2.4H8M12 16v-4h1.1c1.5 0 2.2.8 2.2 2s-.7 2-2.2 2H12M17 16v-4h2" /></>,
  table: <><rect x="4" y="5" width="16" height="14" rx="2" /><path d="M4 10h16M4 15h16M10 5v14M15 5v14" /></>,
  presentation: <><rect x="4" y="4" width="16" height="12" rx="1.8" /><path d="M12 16v4M8 20h8M9 12l2-2 2 1.5 2-3" /></>,
  browser: <><rect x="3.5" y="5" width="17" height="14" rx="2.2" /><path d="M3.5 9h17M7 7h.01M10 7h.01M13 7h.01" /><path d="M8 13h8M8 16h5" /></>,
  pencil: <path d="M4 20h4L19 9a2 2 0 0 0-3-3L5 16v4z" />,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-3.6-3.6" /></>,
  list: <><path d="M8 6h12M8 12h12M8 18h12" /><circle cx="4" cy="6" r="0.6" /><circle cx="4" cy="12" r="0.6" /><circle cx="4" cy="18" r="0.6" /></>,
  sliders: <><path d="M4 7h7M15 7h5M4 17h5M13 17h7" /><circle cx="13" cy="7" r="2" /><circle cx="11" cy="17" r="2" /></>,
  check: <path d="M5 12.5l4.5 4.5L19 7" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  chevronDown: <path d="M6 9l6 6 6-6" />,
  chevronRight: <path d="M9 6l6 6-6 6" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8" /></>,
  moon: <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.6 6.6 0 0 0 9.8 9.8z" />,
  monitor: <><rect x="3" y="4" width="18" height="12.5" rx="2" /><path d="M9 21h6M12 16.5V21" /></>,
  trash: <><path d="M4 7h16M9 7V5a1.5 1.5 0 0 1 1.5-1.5h3A1.5 1.5 0 0 1 15 5v2M6.5 7l.8 12a1.5 1.5 0 0 0 1.5 1.4h6.4a1.5 1.5 0 0 0 1.5-1.4L18.5 7" /></>,
  archive: <><path d="M4.5 6.5h15a1 1 0 0 1 1 1v2.2h-17V7.5a1 1 0 0 1 1-1z" /><path d="M5.5 9.7v8.8a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V9.7" /><path d="M9.5 14h5" /></>,
  sidebar: <><rect x="3" y="4" width="18" height="16" rx="2.2" /><path d="M9 4v16" /></>,
  cpu: <><rect x="7" y="7" width="10" height="10" rx="2" /><path d="M10 3v2M14 3v2M10 19v2M14 19v2M3 10h2M3 14h2M19 10h2M19 14h2" /></>,
  globe: <><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c2.5 2.6 2.5 15.4 0 18M12 3c-2.5 2.6-2.5 15.4 0 18" /></>,
  clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 1.8" /></>,
  refresh: <><path d="M20 12a8 8 0 1 1-2.3-5.6" /><path d="M20 4v4h-4" /></>,
  arrowRight: <path d="M5 12h14M13 6l6 6-6 6" />,
  copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h8" /></>,
  spark: <path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17l-1.9-5.1L4.5 10l5.6-1.4z" />,
  // Pin (VST-040): outline vs filled for un-pinned/pinned thread messages.
  star: <path d="M12 4l2.3 4.9 5.2.6-3.9 3.5 1.1 5.2L12 15.9 7.2 18.7l1.1-5.2-3.9-3.5 5.2-.6z" />,
  starFilled: <path d="M12 4l2.3 4.9 5.2.6-3.9 3.5 1.1 5.2L12 15.9 7.2 18.7l1.1-5.2-3.9-3.5 5.2-.6z" fill="currentColor" />,
  bolt: <path d="M13 2L5 13h6l-1 9 8-12h-6z" />,
  openai: <><path d="M10.7 3.4a4.2 4.2 0 0 1 6.3 3.3 4.2 4.2 0 0 1 3.6 6.1 4.2 4.2 0 0 1-3.7 6 4.2 4.2 0 0 1-6.4 1.9 4.2 4.2 0 0 1-6.2-3.3 4.2 4.2 0 0 1-1-7.8 4.2 4.2 0 0 1 7.4-6.2z" /><path d="M8 8.6l4-2.3 4 2.3v4.7l-4 2.4-4-2.4z" /><path d="M12 6.3v4.6l4 2.4M16 8.6l-4 2.3-4-2.3M8 13.3l4-2.4v4.8" /></>,
  edit: <path d="M4 20h4L19 9a2 2 0 0 0-3-3L5 16v4z" />,
  back: <path d="M19 12H5M11 6l-6 6 6 6" />,
  shield: <path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z" />,
  dots: <><circle cx="5" cy="12" r="1.4" /><circle cx="12" cy="12" r="1.4" /><circle cx="19" cy="12" r="1.4" /></>,
  warning: <><path d="M12 3l9.5 16.5H2.5z" /><path d="M12 10v4M12 17.5v.01" /></>,
  layers: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>,
  branch: <><circle cx="6" cy="6" r="2.3" /><circle cx="6" cy="18" r="2.3" /><circle cx="18" cy="7" r="2.3" /><path d="M6 8.3v7.4M18 9.3c0 3.2-2.4 4.7-5.5 5.3-2 .4-2.5 1.4-2.5 3" /></>,
  message: <path d="M21 11.5a8.4 8.4 0 0 1-8.5 8.5 9 9 0 0 1-4-.9L3.5 20.5l1.4-4.2a8.4 8.4 0 0 1-1-4A8.5 8.5 0 0 1 12.5 3 8.4 8.4 0 0 1 21 11.5z" />,
  grid: <><rect x="4" y="4" width="7" height="7" rx="1.6" /><rect x="13" y="4" width="7" height="7" rx="1.6" /><rect x="4" y="13" width="7" height="7" rx="1.6" /><rect x="13" y="13" width="7" height="7" rx="1.6" /></>,
  mail: <><rect x="3" y="5" width="18" height="14" rx="2.2" /><path d="M3.5 6.8l8.5 5.7 8.5-5.7" /></>,
  plug: <><path d="M9 2v6M15 2v6" /><path d="M7 8h10v3a5 5 0 0 1-10 0z" /><path d="M12 16v6" /></>,
  bell: <><path d="M18 8.5a6 6 0 0 0-12 0c0 6-2.5 7.5-2.5 7.5h17S18 14.5 18 8.5z" /><path d="M10.5 20a2 2 0 0 0 3 0" /></>,
  play: <path d="M7 4.5l12 7.5-12 7.5z" />,
  at: <><circle cx="12" cy="12" r="3.6" /><path d="M15.6 12v1.6a2.4 2.4 0 0 0 4.4 1.3A9 9 0 1 0 16 19.4" /></>,
  attach: <path d="M21 11.5l-8.6 8.6a5 5 0 0 1-7-7L13.8 5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.6 1.6 0 0 1-2.3-2.3l7.9-7.9" />,
}

export type IconName = keyof typeof P

export function Icon({ name, size = 16, className, strokeWidth = 1.7 }: { name: IconName; size?: number; className?: string; strokeWidth?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {P[name]}
    </svg>
  )
}

// TensorFold's three folded lanes, matching the existing app icon.
export function Logo({ size = 22, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" className={className} aria-hidden="true">
      <path d="M12 16h40M12 32h40M12 48h24" stroke="currentColor" strokeWidth="6" strokeLinecap="round" />
    </svg>
  )
}
