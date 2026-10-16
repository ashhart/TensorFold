// Project-local skills (VST-010): repo-authored SKILL.md folders under
// `<project.cwd>/.vontra/skills/<name>/` become per-project skills — UNTRUSTED until the
// user reviews each one (they're repo content: anyone who can commit can write them), and
// hash-pinned so a content change after review drops the skill back out of runs until
// re-trusted. Scanning reuses the hardened agent-skills scanner (realpath-confined,
// bounded reads).

import { createHash } from 'node:crypto'
import path from 'node:path'
import { DEFAULT_AGENT_SKILL_MAX_BYTES, readBoundedText, scanAgentSkills } from '../../agent/agent-skills'
import {
  emptySkillTrustRegistry,
  evaluateSkillTrust,
  normalizeSkillTrustRegistry,
  revokeProjectSkillTrust,
  trustProjectSkill,
  type SkillTrustRegistry,
} from '../../agent/skill-trust'
import type { Project, ProjectSkillView, Skill } from '../../agent/types'

export interface ProjectSkillsRuntimeDeps {
  loadTrust(): Promise<unknown>
  saveTrust(registry: SkillTrustRegistry): Promise<void> | void
}

export const PROJECT_SKILLS_DIR = path.join('.vontra', 'skills')

type ScannableProject = Pick<Project, 'id' | 'cwd' | 'workspace'>

export function createProjectSkillsRuntime(deps: ProjectSkillsRuntimeDeps) {
  let registry: SkillTrustRegistry | null = null
  // Trusted, ready-for-context skills per project — refreshed at run start and on trust
  // changes so the (synchronous) tool-context builder can read it without fs I/O.
  const cache = new Map<string, Skill[]>()

  async function trustRegistry(): Promise<SkillTrustRegistry> {
    if (registry) return registry
    try {
      registry = normalizeSkillTrustRegistry(await deps.loadTrust())
    } catch {
      registry = emptySkillTrustRegistry()
    }
    return registry
  }

  function persist(): void {
    if (!registry) return
    try { void Promise.resolve(deps.saveTrust(registry)).catch(() => {}) } catch { /* best-effort */ }
  }

  /** Live scan of a project's repo skills with their trust states (drives the trust UI). */
  async function scanProjectSkills(project: ScannableProject): Promise<ProjectSkillView[]> {
    // Repo-skill discovery is local-filesystem-only today. Treating an SSH path as a local path
    // lets a same-named local directory inject reviewed instructions into a remote run. Clear any
    // previously trusted cache entry and expose no repo skills until the scanner has an SSH backend.
    if (project.workspace?.kind === 'ssh' || !project.cwd?.trim()) {
      cache.delete(project.id)
      return []
    }
    try {
      const reg = await trustRegistry()
      const found = await scanAgentSkills(path.join(project.cwd, PROJECT_SKILLS_DIR))
      const views: ProjectSkillView[] = []
      for (const meta of found) {
        const text = await readBoundedText(meta.skillPath, DEFAULT_AGENT_SKILL_MAX_BYTES)
        if (text == null) continue
        const contentHash = createHash('sha256').update(text).digest('hex')
        const trust = evaluateSkillTrust(reg, project.id, meta.id, contentHash)
        views.push({
          contentHash,
          trust,
          skill: {
            id: `ps_${project.id}_${meta.id}`,
            name: meta.name,
            description: meta.description,
            body: text,
            // Only reviewed content runs: untrusted and changed-since-review are both OFF.
            enabled: trust === 'trusted',
            scope: 'project',
            projectId: project.id,
            createdAt: meta.updatedAt,
            updatedAt: meta.updatedAt,
          },
        })
      }
      // Refresh the run cache as a side effect of every scan — the UI and the runs always
      // agree on what is trusted.
      cache.set(project.id, views.filter((view) => view.trust === 'trusted').map((view) => view.skill))
      return views
    } catch (error) {
      // A failed refresh must not preserve an older trusted body. Missing/unreadable repo state
      // degrades to no project skills for the next run.
      cache.delete(project.id)
      throw error
    }
  }

  /** Trusted project skills for the tool context (synchronous — served from the cache). */
  function cachedProjectSkills(projectId: string | null | undefined, workspace?: Project['workspace']): Skill[] {
    // Defense in depth at the synchronous context-assembly boundary: even if a caller
    // missed the awaited refresh, an SSH run cannot consume a previously cached local skill.
    if (workspace?.kind === 'ssh') {
      if (projectId) cache.delete(projectId)
      return []
    }
    return projectId ? cache.get(projectId) ?? [] : []
  }

  /** Refresh the cache before a run so a freshly-edited repo skill can't ride a stale trust. */
  async function refreshProjectSkills(project: ScannableProject | null | undefined): Promise<void> {
    if (!project) return
    try { await scanProjectSkills(project) } catch { cache.delete(project.id) }
  }

  /** Trust or revoke one project skill; trust pins the CURRENT content hash. */
  async function setProjectSkillTrust(project: ScannableProject, skillId: string, trusted: boolean): Promise<ProjectSkillView[]> {
    if (project.workspace?.kind === 'ssh') {
      cache.delete(project.id)
      throw new Error('Repo-authored skills are unavailable for SSH projects until remote skill scanning is supported.')
    }
    const reg = await trustRegistry()
    const prefix = `ps_${project.id}_`
    const bareId = skillId.startsWith(prefix) ? skillId.slice(prefix.length) : skillId
    if (!trusted) {
      registry = revokeProjectSkillTrust(reg, project.id, bareId)
      persist()
      return scanProjectSkills(project)
    }
    // Trust applies to what is on disk RIGHT NOW — rescan for the live hash first.
    const views = await scanProjectSkills(project)
    const target = views.find((view) => view.skill.id === `${prefix}${bareId}`)
    if (!target) throw new Error('project skill not found')
    registry = trustProjectSkill(reg, { projectId: project.id, skillId: bareId, contentHash: target.contentHash })
    persist()
    return scanProjectSkills(project)
  }

  return { scanProjectSkills, cachedProjectSkills, refreshProjectSkills, setProjectSkillTrust }
}

export type ProjectSkillsRuntime = ReturnType<typeof createProjectSkillsRuntime>
