import { assertSavedMemoryRead, effectiveThreadMemoryPolicy } from '../../shared/thread-memory-policy'
import { botToolBlockReason } from '../../shared/bot-tool-policy'
import path from 'node:path'
import { buildAssistantDoctorReport, formatAssistantDoctorReport } from './assistant-doctor'
import { buildSkillCurationReport, formatSkillCurationReport } from './skill-curator'
import { buildCodebaseMap, buildSymbolIndex, formatCodebaseMap, formatRewritePreview, formatSafeRefactorPreview, formatSecurityScan, formatStructuralSearch, formatSymbolIndex, rewritePreview, safeRefactorPreview, semgrepCompatibleScan, structuralSearch } from './code-intelligence'
import { buildWorkspaceRetrievalIndex, filterRetrievalProjectMemorySources, filterRetrievalThreadSources, formatRetrievalSearchResult, searchRetrievalIndex } from './retrieval-index'
import { canRecallThread, formatSessionReadResult, formatSessionSearchResult, readSessionThread, searchSessionThreads } from './session-recall'
import { buildDeferredToolCatalog, findDeferredTool, formatDeferredToolDescription, formatToolCatalogSearch, searchDeferredToolCatalog } from './tool-disclosure'
import { filterNonSensitiveRetrievalKinds } from './tool-visualization'
import type { ToolContext, ToolResult } from './tool-context'


// Retrieval and code intelligence revalidate live privacy before exposing source evidence.
export async function executeRetrievalTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'retrieval_search': {
      const sensitiveAllowed = ctx.sensitiveRetrievalAllowed !== false
      const requestedKinds = Array.isArray(args.kinds) ? args.kinds : typeof args.kinds === 'string' ? args.kinds : undefined
      const effectiveKinds = sensitiveAllowed ? requestedKinds : filterNonSensitiveRetrievalKinds(requestedKinds)
      const collected = await buildWorkspaceRetrievalIndex({
        cwd: ctx.cwd,
        savedMemoryAllowed: effectiveThreadMemoryPolicy(ctx.currentThread ?? {}, ctx.memoryPolicy).useExisting,
        currentThreadId: ctx.currentThread?.id,
        path: typeof args.path === 'string' ? args.path : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        kinds: effectiveKinds,
        maxCodeFiles: Number.isFinite(Number(args.max_code_files)) ? Number(args.max_code_files) : undefined,
        maxConnectorItems: Number.isFinite(Number(args.max_connector_items)) ? Number(args.max_connector_items) : undefined,
        emailSettings: sensitiveAllowed ? ctx.emailSettings : undefined,
        slackSettings: sensitiveAllowed ? ctx.slackSettings : undefined,
        calendarSettings: sensitiveAllowed ? ctx.calendarSettings : undefined,
        documentSettings: sensitiveAllowed ? ctx.documentSettings : undefined,
        browserSettings: sensitiveAllowed ? ctx.browserSettings : undefined,
        connectorEvents: sensitiveAllowed ? ctx.connectorEvents : undefined,
        threads: sensitiveAllowed ? ctx.threads : undefined,
        recallThreads: sensitiveAllowed ? ctx.recallThreads : undefined,
        missions: sensitiveAllowed ? ctx.missions : undefined,
        rooms: sensitiveAllowed ? ctx.rooms : undefined,
        tickets: sensitiveAllowed ? ctx.tickets : undefined,
        projectMemory: sensitiveAllowed ? ctx.projectMemory : undefined,
        recallProjectMemory: sensitiveAllowed ? ctx.recallProjectMemory : undefined,
      })
      const visible = filterRetrievalThreadSources(collected, ctx.recallThreads?.() ?? ctx.threads ?? [], ctx.currentThread?.id, ctx.recallMissions?.() ?? ctx.missions)
      const index = filterRetrievalProjectMemorySources(visible, sensitiveAllowed ? ctx.recallProjectMemory?.() ?? ctx.projectMemory ?? [] : [])
      const result = searchRetrievalIndex(index, {
        query: typeof args.query === 'string' ? args.query : undefined,
        kinds: effectiveKinds,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
      })
      ctx.recordRecallSources?.([...new Set(result.hits.flatMap(({ record }) => [record.id, ...record.sourceIds]).filter((id) => id.startsWith('thread:')).map((id) => id.split(':')[1]))])
      ctx.assertRecallSourcesAllowed?.()
      const output = formatRetrievalSearchResult(result)
      const dlpNotice = sensitiveAllowed ? '' : `Sensitive retrieval sources omitted (${ctx.sensitiveRetrievalReason || 'policy disabled connector, chat, document, and memory retrieval for this run'}). Only code and artifact records were indexed.\n\n`
      return {
        ok: true,
        output: dlpNotice + output,
        title: 'Retrieval search',
        meta: `${result.hits.length} hits · ${index.records.length} indexed${sensitiveAllowed ? '' : ' · sensitive sources omitted'}`,
      }
    }

    case 'session_search': {
      assertSavedMemoryRead(ctx.currentThread ?? {}, ctx.memoryPolicy)
      const options = {
        query: String(args.query ?? ''),
        projectId: typeof args.projectId === 'string' && args.projectId.trim() ? args.projectId.trim() : undefined,
        currentThreadId: ctx.currentThread?.id,
        includeCurrent: args.includeCurrent === true,
        includeArchived: args.includeArchived === true,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
      }
      const found = ctx.searchSessions
        ? await ctx.searchSessions(options)
        : searchSessionThreads(ctx.threads ?? [], options)
      const visibleIds = new Set((ctx.recallThreads?.() ?? ctx.threads ?? []).filter((thread) => canRecallThread(thread, ctx.currentThread?.id)).map((thread) => thread.id))
      const matches = found.matches.filter((match) => visibleIds.has(match.threadId))
      const result = { ...found, matches, omitted: matches.length === found.matches.length ? found.omitted : 0 }
      ctx.recordRecallSources?.(matches.map((match) => match.threadId))
      ctx.assertRecallSourcesAllowed?.()
      return {
        ok: true,
        output: formatSessionSearchResult(result),
        title: 'Session search',
        meta: `${result.matches.length} matches${result.omitted ? ` · ${result.omitted} omitted` : ''}`,
      }
    }

    case 'session_read': {
      assertSavedMemoryRead(ctx.currentThread ?? {}, ctx.memoryPolicy)
      const result = readSessionThread(ctx.threads ?? [], {
        threadId: String(args.threadId ?? ''),
        currentThreadId: ctx.currentThread?.id,
        maxMessages: Number.isFinite(Number(args.maxMessages)) ? Number(args.maxMessages) : undefined,
      })
      if (result.thread) ctx.recordRecallSources?.([result.thread.id])
      ctx.assertRecallSourcesAllowed?.()
      return {
        ok: !!result.thread,
        output: formatSessionReadResult(result),
        title: 'Session read',
        meta: result.thread ? `${result.messages.length} messages${result.omittedBefore ? ` · ${result.omittedBefore} earlier omitted` : ''}` : 'not found',
        path: result.thread?.id,
      }
    }

    case 'assistant_doctor': {
      const report = buildAssistantDoctorReport({
        currentThread: ctx.currentThread,
        threads: ctx.threads,
        mcpTools: ctx.mcpTools,
        skills: ctx.skills,
        commands: ctx.commands,
        localScripts: ctx.localScripts,
        packagedPlugins: ctx.packagedPlugins,
        disabledCapabilityIds: ctx.disabledCapabilityIds,
      })
      return {
        ok: true,
        output: formatAssistantDoctorReport(report),
        title: 'Assistant doctor',
        meta: report.promptSnapshot.present ? `prompt ${report.promptSnapshot.fingerprint}` : 'prompt snapshot missing',
      }
    }

    case 'skill_curator': {
      const report = buildSkillCurationReport({
        skills: ctx.skills,
        commands: ctx.commands,
        pendingSuggestions: ctx.skillSuggestions,
        staleAfterDays: Number.isFinite(Number(args.stale_after_days)) ? Number(args.stale_after_days) : undefined,
      })
      return {
        ok: true,
        output: formatSkillCurationReport(report),
        title: 'Skill curator',
        meta: `${report.findings.length} findings`,
      }
    }

    case 'tool_search': {
      const catalog = buildDeferredToolCatalog((ctx.mcpTools ?? []).filter((tool) => !botToolBlockReason(ctx.currentThread?.bot, tool.callName, {}, ctx.allowedToolNames)))
      const matches = searchDeferredToolCatalog(catalog, {
        query: String(args.query ?? ''),
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
      })
      const result = {
        query: String(args.query ?? '').trim(),
        matches,
        omitted: Math.max(0, catalog.tools.length - matches.length),
      }
      return {
        ok: true,
        output: formatToolCatalogSearch(result),
        title: 'Tool search',
        meta: `${matches.length} matches · ${catalog.tools.length} deferred`,
      }
    }

    case 'tool_describe': {
      const catalog = buildDeferredToolCatalog((ctx.mcpTools ?? []).filter((tool) => !botToolBlockReason(ctx.currentThread?.bot, tool.callName, {}, ctx.allowedToolNames)))
      const tool = findDeferredTool(catalog, String(args.name ?? ''))
      return {
        ok: !!tool,
        output: formatDeferredToolDescription(tool),
        title: 'Tool describe',
        meta: tool?.name ?? 'not found',
      }
    }

    case 'tool_call': {
      const name = String(args.name ?? '').trim()
      const catalog = buildDeferredToolCatalog((ctx.mcpTools ?? []).filter((tool) => !botToolBlockReason(ctx.currentThread?.bot, tool.callName, {}, ctx.allowedToolNames)))
      const tool = findDeferredTool(catalog, name)
      if (!tool) return { ok: false, output: `No deferred tool found named "${name}". Use tool_search first.`, title: 'Tool call' }
      if (!name.startsWith('mcp__')) return { ok: false, output: 'Only deferred MCP tools can be called through tool_call.', title: 'Tool call' }
      if (!ctx.callMcpTool) return { ok: false, output: 'MCP host is not configured.', title: 'Tool call' }
      const callArgs = args.arguments && typeof args.arguments === 'object' && !Array.isArray(args.arguments)
        ? args.arguments as Record<string, any>
        : {}
      const result = await ctx.callMcpTool(name, callArgs)
      return { ok: result.ok, output: result.output, title: result.title ?? 'Tool call', meta: result.meta ?? name }
    }

    case 'ast_grep_search': {
      const result = await structuralSearch(ctx.cwd, {
        pattern: String(args.pattern ?? ''),
        path: typeof args.path === 'string' ? args.path : undefined,
        maxMatches: Number.isFinite(Number(args.max_matches)) ? Number(args.max_matches) : undefined,
      })
      return {
        ok: true,
        output: formatStructuralSearch(result),
        title: 'Structural search',
        meta: `${result.matches.length}${result.truncated ? '+' : ''} matches`,
      }
    }

    case 'ast_grep_rewrite_preview': {
      const result = await rewritePreview(ctx.cwd, {
        pattern: String(args.pattern ?? ''),
        rewrite: String(args.rewrite ?? ''),
        path: typeof args.path === 'string' ? args.path : undefined,
        maxReplacements: Number.isFinite(Number(args.max_replacements)) ? Number(args.max_replacements) : undefined,
      })
      return {
        ok: true,
        output: formatRewritePreview(result),
        title: 'Rewrite preview',
        diff: result.diff,
        meta: `${result.files.length} files · +${result.added} −${result.removed} · not applied`,
      }
    }

    case 'semgrep_scan': {
      const result = await semgrepCompatibleScan(ctx.cwd, {
        path: typeof args.path === 'string' ? args.path : undefined,
        autofix: Boolean(args.autofix),
        maxFindings: Number.isFinite(Number(args.max_findings)) ? Number(args.max_findings) : undefined,
      })
      const blocking = result.findings.filter((finding) => finding.blocking).length
      return {
        ok: blocking === 0,
        output: formatSecurityScan(result),
        title: 'Semgrep scan',
        diff: result.diff,
        meta: `${result.findings.length} findings · ${blocking} blocking · ${result.proposedFixes.length} fixes`,
      }
    }

    case 'codebase_map': {
      const result = await buildCodebaseMap(ctx.cwd, {
        path: typeof args.path === 'string' ? args.path : undefined,
        maxFiles: Number.isFinite(Number(args.max_files)) ? Number(args.max_files) : undefined,
      })
      return {
        ok: true,
        output: formatCodebaseMap(result),
        title: 'Codebase map',
        meta: `${result.filesScanned} files · ${result.imports.length} imports · ${result.symbols.length} symbols`,
      }
    }

    case 'symbol_index': {
      const result = await buildSymbolIndex(ctx.cwd, {
        path: typeof args.path === 'string' ? args.path : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        maxFiles: Number.isFinite(Number(args.max_files)) ? Number(args.max_files) : undefined,
        maxSymbols: Number.isFinite(Number(args.max_symbols)) ? Number(args.max_symbols) : undefined,
      })
      return {
        ok: true,
        output: formatSymbolIndex(result),
        title: 'Symbol index',
        meta: `${result.filesScanned} files · ${result.symbols.length}${result.truncated ? '+' : ''} symbols`,
      }
    }

    case 'safe_refactor_preview': {
      const result = await safeRefactorPreview(ctx.cwd, {
        pattern: String(args.pattern ?? ''),
        rewrite: String(args.rewrite ?? ''),
        path: typeof args.path === 'string' ? args.path : undefined,
        maxReplacements: Number.isFinite(Number(args.max_replacements)) ? Number(args.max_replacements) : undefined,
      })
      const blocking = result.scan.findings.filter((finding) => finding.blocking).length
      return {
        ok: blocking === 0,
        output: formatSafeRefactorPreview(result),
        title: 'Safe refactor preview',
        diff: result.rewrite.diff,
        meta: `${result.search.matches.length} matches · ${result.rewrite.files.length} files · ${blocking} blocking · not applied`,
      }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
