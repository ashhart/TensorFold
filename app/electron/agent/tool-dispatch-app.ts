import { assertMemoryContribution } from '../../shared/thread-memory-policy'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { listSkills, viewSkill, renderSkillView } from './skill-access'
import { redactSecretValues } from './secret-detection'
import { SECRET_MASK } from './security'
import { SKIP_DIRS } from './tool-patch'
import { normalizeVisualizationSpec } from './tool-visualization'
import { resolveWorkspacePath, workspaceBackend } from './tool-workspace'
import type { ToolContext, ToolResult } from './tool-context'

function isHomeFolder(folder: string): boolean {
  const home = path.resolve(os.homedir())
  const target = path.resolve(folder)
  return target === home
}

// App actions, reusable skills and artifact registration keep their existing consent boundaries.
export async function executeAppTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'create_project': {
      const projectName = String(args.name ?? '').trim()
      if (!projectName) return { ok: false, output: 'A project name is required.' }
      const folder = args.folder ? String(args.folder).trim() : undefined
      if (folder) {
        if (!folder.startsWith('/')) return { ok: false, output: 'Project folder must be an absolute path (starting with "/"), or omitted.' }
        // The folder must exist and must not be the user's home directory — a
        // project writes AGENTS.md/CLAUDE.md into its cwd, so never let the model
        // point that at $HOME or a nonexistent path. Full validation (home-folder
        // guard incl. symlink resolution) runs again server-side on projects:create.
        try {
          const stat = await fs.stat(folder)
          if (!stat.isDirectory()) return { ok: false, output: `Project folder is not a directory: ${folder}` }
        } catch {
          return { ok: false, output: `Project folder does not exist: ${folder}` }
        }
        if (isHomeFolder(folder)) return { ok: false, output: 'Refusing to use your home folder as a project workspace. Pick a project subfolder instead.' }
      }
      return { ok: true, title: 'Create project', meta: projectName, output: `Creating project “${projectName}”${folder ? ` in ${folder}` : ''}.`, appAction: { kind: 'create_project', payload: { name: projectName, folder } } }
    }

    case 'create_room': {
      const roomName = String(args.name ?? '').trim() || 'New room'
      return { ok: true, title: 'Create room', meta: roomName, output: `Creating room “${roomName}”.`, appAction: { kind: 'create_room', payload: { name: roomName, projectId: args.projectId ? String(args.projectId) : undefined } } }
    }

    case 'create_kanban_ticket': {
      const ticketTitle = String(args.title ?? '').trim()
      if (!ticketTitle) return { ok: false, output: 'A ticket title is required.' }
      const TYPES = ['epic', 'task', 'subticket', 'bug'], PRIORITIES = ['low', 'medium', 'high', 'urgent']
      const type = args.type != null ? String(args.type) : undefined
      const priority = args.priority != null ? String(args.priority) : undefined
      if (type && !TYPES.includes(type)) return { ok: false, output: `Invalid ticket type "${type}". Use one of: ${TYPES.join(', ')}.` }
      if (priority && !PRIORITIES.includes(priority)) return { ok: false, output: `Invalid priority "${priority}". Use one of: ${PRIORITIES.join(', ')}.` }
      return { ok: true, title: 'Add kanban ticket', meta: ticketTitle, output: `Adding ticket “${ticketTitle}” to the board.`, appAction: { kind: 'create_kanban', payload: { title: ticketTitle, body: args.body ? String(args.body) : '', type, priority, projectId: args.projectId ? String(args.projectId) : undefined } } }
    }

    case 'open_view': {
      const VIEWS = ['home', 'projects', 'settings', 'plugins', 'automations', 'commandCentre', 'assistant']
      const view = String(args.view ?? '').trim()
      if (!VIEWS.includes(view)) return { ok: false, output: `Unknown view "${view}". Use one of: ${VIEWS.join(', ')}.` }
      return { ok: true, title: 'Open', meta: view, output: `Opening ${view}.`, appAction: { kind: 'open_view', payload: { view } } }
    }

    case 'change_setting': {
      const SETTING_VALUES: Record<string, string[]> = { theme: ['system', 'light', 'dark'], approvalMode: ['suggest', 'auto', 'readonly'], reasoningEffort: ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512'] }
      const key = String(args.key ?? '').trim()
      const value = args.value != null ? String(args.value).trim() : ''
      if (!SETTING_VALUES[key]) return { ok: false, output: `Unknown setting "${key}". Use one of: ${Object.keys(SETTING_VALUES).join(', ')}.` }
      if (!SETTING_VALUES[key].includes(value)) return { ok: false, output: `Invalid value "${value}" for ${key}. Use one of: ${SETTING_VALUES[key].join(', ')}.` }
      return { ok: true, title: 'Change setting', meta: key, output: `Proposed: set ${key} to ${value}. Waiting for you to confirm.`, appAction: { kind: 'change_setting', payload: { key, value }, state: 'pending' } }
    }

    case 'suggest_skill': {
      // Propose (never create) a reusable Skill or Command. The user reviews + accepts.
      const kind = String(args.kind ?? 'skill').trim() === 'command' ? 'command' : 'skill'
      const name = String(args.name ?? '').trim()
      const body = String(args.body ?? '').trim()
      if (!name || !body) return { ok: false, output: 'suggest_skill needs a name and a body.' }
      assertMemoryContribution(ctx.currentThread ?? {}, ctx.memoryPolicy)
      if (!ctx.fileSuggestion) return { ok: true, title: 'Suggest', meta: name, output: 'Suggestions are unavailable here.' }
      ctx.fileSuggestion({
        kind,
        name,
        body,
        description: args.description != null ? String(args.description) : '',
        argHint: args.argHint != null ? String(args.argHint) : undefined,
        rationale: args.rationale != null ? String(args.rationale) : '',
      })
      return { ok: true, title: kind === 'command' ? 'Suggest command' : 'Suggest skill', meta: name, output: `Filed a ${kind} suggestion "${name}". The user can review and save it from Skills & Commands — nothing was created.` }
    }

    case 'skills_list': {
      const list = listSkills(ctx.skills)
      if (!list.length) return { ok: true, title: 'Skills', output: 'No saved skills.' }
      return { ok: true, title: 'Skills', meta: `${list.length}`, output: list.map((s) => `- ${s.name}: ${s.description || '(no description)'}`).join('\n') }
    }

    case 'skill_view': {
      const skill = viewSkill(ctx.skills, String(args.name ?? ''))
      if (!skill) return { ok: false, output: `No enabled skill named "${args.name}". Use skills_list to see available skills.` }
      return { ok: true, title: 'Skill', meta: skill.name, output: renderSkillView(skill) }
    }

    case 'save_artifact': {
      if (!ctx.registerArtifact || !ctx.currentThread) return { ok: false, output: 'Artifact registration is not available in this run context.', meta: 'unavailable' }
      const title = String(args.title ?? '').trim()
      if (!title) return { ok: false, output: 'save_artifact requires a title.', meta: 'invalid' }
      const hasPath = typeof args.path === 'string' && args.path.trim().length > 0
      const hasContent = typeof args.content === 'string' && args.content.length > 0
      if (!hasPath && !hasContent) return { ok: false, output: 'Provide the path of an existing file OR inline content.', meta: 'invalid' }
      if (hasPath && hasContent) return { ok: false, output: 'Provide a path OR content, not both.', meta: 'invalid' }
      if (hasPath && (ctx.workspace?.kind === 'ssh' || ctx.backend?.kind === 'ssh')) {
        return {
          ok: false,
          output: 'Path-backed artifacts from SSH workspaces cannot be previewed or exported safely yet. Save bounded inline content, or copy the output into a managed local artifact first.',
          meta: 'remote_unsupported',
        }
      }
      let filePath: string | null = null
      let sizeBytes: number | null = null
      if (hasPath) {
        // Same confinement as read_file: the path must resolve inside the workspace,
        // and the file must actually exist (an artifact is evidence, not a promise). Use
        // metadata for the size: reading the entire file merely to register it defeats the
        // bounded preview/export path and lets a huge artifact exhaust local or SSH memory.
        filePath = resolveWorkspacePath(ctx, args.path)
        try {
          const info = await workspaceBackend(ctx).stat(filePath)
          if (!info.isFile || !Number.isSafeInteger(info.size) || info.size < 0) throw new Error('not a regular file')
          sizeBytes = info.size
        } catch {
          return { ok: false, output: `Cannot inspect ${args.path} — the artifact must be an existing regular file.`, meta: 'missing' }
        }
      }
      const artifact = ctx.registerArtifact({
        title,
        kind: typeof args.kind === 'string' ? args.kind : null,
        path: filePath,
        inlineText: hasContent ? redactSecretValues(String(args.content), SECRET_MASK).text : null,
        sizeBytes,
        provenance: {
          threadId: ctx.currentThread.id,
          tool: 'save_artifact',
          workspace: ctx.workspace ?? { kind: 'local', path: ctx.cwd },
          sourceRef: typeof args.source_ref === 'string' && args.source_ref.trim() ? args.source_ref.trim() : null,
        },
      })
      return { ok: true, output: `Registered artifact "${artifact.title}" (${artifact.kind}).`, title: `Artifact: ${artifact.title}`, meta: artifact.kind }
    }

    case 'visualize': {
      const visualization = normalizeVisualizationSpec(args)
      const counts = [
        visualization.nodes?.length ? `${visualization.nodes.length} nodes` : '',
        visualization.bars?.length ? `${visualization.bars.length} bars` : '',
        visualization.metrics?.length ? `${visualization.metrics.length} metrics` : '',
        visualization.events?.length ? `${visualization.events.length} events` : '',
      ].filter(Boolean).join(' · ')
      return {
        ok: true,
        output: `Rendered ${visualization.kind} visualization "${visualization.title}".`,
        title: visualization.title,
        visualization,
        meta: counts || visualization.kind,
      }
    }

    case 'repo_wiki': {
      const root = ctx.cwd
      const out: string[] = []
      try {
        const pkg = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8'))
        out.push(`# ${pkg.name ?? path.basename(root)}`)
        if (pkg.description) out.push(`\n${pkg.description}`)
        const scripts = Object.entries(pkg.scripts ?? {})
        if (scripts.length) { out.push('\n## Scripts'); for (const [k, v] of scripts) out.push(`- \`npm run ${k}\`: ${v}`) }
        const deps = Object.keys(pkg.dependencies ?? {})
        if (deps.length) out.push(`\n## Key dependencies\n${deps.slice(0, 24).join(', ')}`)
      } catch { out.push(`# ${path.basename(root)}`) }

      const KNOWN = ['index.html', 'vite.config.ts', 'vite.config.js', 'tsconfig.json', 'next.config.js', 'src/main.tsx', 'src/main.ts', 'src/index.tsx', 'src/App.tsx', 'electron/main.ts', 'server.ts', 'Dockerfile', 'docker-compose.yml', 'AGENTS.md', 'CLAUDE.md', 'README.md']
      const present: string[] = []
      for (const f of KNOWN) { try { await fs.access(path.join(root, f)); present.push(f) } catch { /* absent */ } }
      if (present.length) out.push(`\n## Entry points & config\n${present.map((f) => `- ${f}`).join('\n')}`)

      out.push('\n## Structure')
      const tree: string[] = []
      const walk = async (dir: string, prefix: string, depth: number) => {
        if (depth > 2 || tree.length > 200) return
        let entries: any[] = []
        try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
        const filtered = entries
          .filter((e) => !e.name.startsWith('.') && !SKIP_DIRS.has(e.name))
          .sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
        for (const e of filtered.slice(0, 40)) {
          tree.push(`${prefix}${e.name}${e.isDirectory() ? '/' : ''}`)
          if (e.isDirectory()) await walk(path.join(dir, e.name), prefix + '  ', depth + 1)
        }
      }
      await walk(root, '', 0)
      out.push(tree.join('\n'))
      return { ok: true, output: out.join('\n'), title: 'Repo map', meta: `${present.length} key files` }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
