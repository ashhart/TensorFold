import path from 'node:path'
import { computeDiff, diffStat } from './diff'
import { blockedReadReason, blockedWriteReason } from './command-policy'
import { commandOutputWithRedactionNotice } from './command-output-redaction'
import { redactSecretValues } from './secret-detection'
import { SECRET_MASK } from './security'
import { checkEditedSyntax, isGatedCodePath, syntaxGateMessage } from './edit-syntax-gate'
import { rel } from './tool-paths'
import { collectFiles, combinedDiff, DEFAULT_GLOB_RESULTS, MAX_GLOB_RESULTS, parseApplyPatchText, patchSummary, replaceLiteral, replacementHunksFromArgs, SKIP_DIRS, stagePatchOperations, stageReplacementHunks, validateRelativePattern, writeAppliedChanges } from './tool-patch'
import { isManagedOutputRef } from './context-governance'
import { boundedRegexSearch } from './bounded-regex-search'
import { workspaceContentSha256 } from './atomic-file-write'
import { isRemote, relWorkspace, resolveWorkspacePath, shArg, workspaceBackend } from './tool-workspace'
import type { ToolContext, ToolResult } from './tool-context'

const READ_FILE_MAX_CHARS = 48_000

// Workspace reads and atomic edits share the same local or SSH backend policies.
export async function executeFileTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'list_dir': {
      const dir = resolveWorkspacePath(ctx, args.path)
      const entries = await workspaceBackend(ctx).readdir(dir)
      const sorted = entries
        .filter((e) => !e.name.startsWith('.'))
        .sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b.name))
      const list = sorted.map((e) => (e.isDirectory ? `${e.name}/` : e.name)).join('\n')
      return { ok: true, output: list || '(empty)', title: relWorkspace(ctx, dir), path: relWorkspace(ctx, dir), meta: `${sorted.length} items` }
    }

    case 'read_file': {
      const requestedPath = String(args.path ?? '')
      const managed = isManagedOutputRef(requestedPath)
      if (!managed) {
        const readBlock = blockedReadReason(requestedPath)
        if (readBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${readBlock}.`, meta: 'blocked' }
      }
      const file = managed ? requestedPath : resolveWorkspacePath(ctx, requestedPath)
      const offsetValue = Number(args.offset ?? 1)
      const limitValue = Number(args.limit ?? 2000)
      const offset = Math.max(1, Number.isFinite(offsetValue) ? Math.trunc(offsetValue) : 1)
      const limit = Math.max(1, Number.isFinite(limitValue) ? Math.trunc(limitValue) : 2000)
      const backend = managed ? null : workspaceBackend(ctx)
      if (backend?.readFileRange) {
        const range = await backend.readFileRange(file, { offset, limit, maxChars: READ_FILE_MAX_CHARS, signal: ctx.signal })
        const note = !range.bounded ? '' : range.longLine
          ? `\n[read_file bounded by TensorFold: line ${offset} alone is longer than the output limit; showing its first ${READ_FILE_MAX_CHARS.toLocaleString()} chars. Use the search tool to find what you need inside it.]`
          : `\n[read_file bounded by TensorFold: showing ${range.lines} lines within the output limit. Continue with offset=${range.nextOffset}, or use a smaller limit.]`
        const redacted = redactSecretValues(range.text + note, SECRET_MASK)
        const count = redacted.hits.length
        const label = relWorkspace(ctx, file)
        return {
          ok: true, output: commandOutputWithRedactionNotice(redacted.text, count), title: label, path: label,
          meta: `${range.totalLines ?? `at least ${offset + range.lines - 1}`} lines${range.bounded ? ' · bounded' : ''}${count ? ` · redacted=${count}` : ''}`,
        }
      }
      const raw = managed ? await ctx.readManagedOutput?.(requestedPath) : await backend!.readFile(file, { signal: ctx.signal })
      if (raw == null) return { ok: false, output: 'Managed output was not found or is no longer available.', title: requestedPath, path: requestedPath, meta: 'missing' }
      const allLines = raw.split('\n')
      const slice = allLines.slice(offset - 1, offset - 1 + limit)
      const numbered = slice.map((l, i) => `${String(offset + i).padStart(5)}\t${l}`).join('\n')
      // Char bound (VST-016): the line limit alone cannot bound a file with enormous lines
      // (minified bundles, single-line JSON). Cut at the last whole line under the budget —
      // or mid-line when even ONE line exceeds it — and say exactly how to continue.
      if (numbered.length > READ_FILE_MAX_CHARS) {
        const cutAt = numbered.lastIndexOf('\n', READ_FILE_MAX_CHARS)
        const kept = cutAt > 0 ? numbered.slice(0, cutAt) : numbered.slice(0, READ_FILE_MAX_CHARS)
        const shownLines = cutAt > 0 ? kept.split('\n').length : 1
        const note = cutAt > 0
          ? `[read_file bounded by TensorFold: this range is ${numbered.length.toLocaleString()} chars; showing ${shownLines} of the ${slice.length} lines requested (file has ${allLines.length} lines). Continue with offset=${offset + shownLines}, or use a smaller limit.]`
          : `[read_file bounded by TensorFold: line ${offset} alone is ${numbered.length.toLocaleString()}+ chars (file has ${allLines.length} lines); showing its first ${READ_FILE_MAX_CHARS.toLocaleString()} chars. Use the search tool to find what you need inside it.]`
        const label = managed ? requestedPath : relWorkspace(ctx, file)
        const redactedBounded = redactSecretValues(`${kept}\n${note}`, SECRET_MASK)
        const boundedCount = managed ? 0 : redactedBounded.hits.length
        return {
          ok: true,
          output: commandOutputWithRedactionNotice(redactedBounded.text, boundedCount),
          title: label,
          path: label,
          meta: `${allLines.length} lines · bounded${boundedCount ? ` · redacted=${boundedCount}` : ''}`,
        }
      }
      const label = managed ? requestedPath : relWorkspace(ctx, file)
      const redacted = redactSecretValues(numbered, SECRET_MASK)
      const count = managed ? 0 : redacted.hits.length
      return {
        ok: true,
        output: commandOutputWithRedactionNotice(redacted.text, count),
        title: label,
        path: label,
        meta: `${allLines.length} lines${count ? ` · redacted=${count}` : ''}`,
      }
    }

    case 'write_file': {
      const backend = workspaceBackend(ctx)
      const file = resolveWorkspacePath(ctx, args.path)
      const writeBlock = blockedWriteReason(relWorkspace(ctx, file))
      if (writeBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${writeBlock}.`, meta: 'blocked' }
      let before = ''
      let existed = false
      try {
        before = await backend.readFile(file, { signal: ctx.signal })
        existed = true
      } catch (error) {
        // A read failure for an existing path is not proof that the path is new.
        // Fail closed instead of replacing a permission error or raced entry.
        if (await backend.exists(file)) throw error
      }
      const newContent = args.content ?? ''
      // Syntax gate: never overwrite an EXISTING code file with something that no longer
      // parses (hand the error back so the model self-corrects). A brand-new file is only
      // flagged, not blocked — scaffolding can legitimately be written in pieces. Fail-open.
      let gateNote = ''
      if (isGatedCodePath(file) && newContent) {
        const syn = await checkEditedSyntax(file, newContent)
        if (!syn.ok) {
          if (existed) return { ok: false, output: syntaxGateMessage(relWorkspace(ctx, file), syn.error || 'parse error'), title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
          gateNote = `\n⚠ Heads up: this new file may have a syntax error — ${syn.error}`
        }
      }
      await backend.writeFile(file, newContent, existed
        ? { expectedSha256: workspaceContentSha256(before) }
        : { ifExists: 'error' })
      const diff = computeDiff(before, newContent)
      const { added, removed } = diffStat(diff)
      return {
        ok: true,
        output: `${existed ? `Updated ${relWorkspace(ctx, file)}` : `Created ${relWorkspace(ctx, file)}`}${gateNote}`,
        title: relWorkspace(ctx, file),
        path: relWorkspace(ctx, file),
        diff,
        changedPaths: [relWorkspace(ctx, file)],
        meta: `+${added} −${removed}`,
      }
    }

    case 'edit_file': {
      const backend = workspaceBackend(ctx)
      const file = resolveWorkspacePath(ctx, args.path)
      const writeBlock = blockedWriteReason(relWorkspace(ctx, file))
      if (writeBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${writeBlock}.`, meta: 'blocked' }
      const before = await backend.readFile(file, { signal: ctx.signal })
      const oldS = args.old_string ?? ''
      const count = oldS ? before.split(oldS).length - 1 : 0
      if (count === 0) return { ok: false, output: `String not found in ${relWorkspace(ctx, file)}.`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
      if (count > 1) return { ok: false, output: `String is not unique (${count} matches) in ${relWorkspace(ctx, file)}.`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
      const after = replaceLiteral(before, oldS, args.new_string ?? '')
      // Syntax gate: edit_file always modifies an existing file, so a broken result is a
      // mistake — block and report it rather than persisting unparseable code.
      if (isGatedCodePath(file)) {
        const syn = await checkEditedSyntax(file, after)
        if (!syn.ok) return { ok: false, output: syntaxGateMessage(relWorkspace(ctx, file), syn.error || 'parse error'), title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
      }
      await backend.writeFile(file, after, { expectedSha256: workspaceContentSha256(before) })
      const diff = computeDiff(before, after)
      const { added, removed } = diffStat(diff)
      return { ok: true, output: `Edited ${relWorkspace(ctx, file)}`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file), diff, changedPaths: [relWorkspace(ctx, file)], meta: `+${added} −${removed}` }
    }

    case 'apply_patch': {
      const changes = typeof args.patch === 'string' && args.patch.trim()
        ? await stagePatchOperations(ctx.cwd, parseApplyPatchText(args.patch))
        : await stageReplacementHunks(ctx.cwd, replacementHunksFromArgs(args))
      // Syntax gate (all-or-nothing): if any hunk would leave an EXISTING code file
      // unparseable, write NONE of them and report — so a multi-file patch can't half-apply
      // a broken edit. New files added by the patch (no prior content) are not blocked.
      for (const ch of changes) {
        if (ch.kind === 'delete' || ch.kind === 'move' || !ch.before) continue
        if (!isGatedCodePath(ch.finalPath) || !ch.after) continue
        const syn = await checkEditedSyntax(ch.finalPath, ch.after)
        if (!syn.ok) return { ok: false, output: syntaxGateMessage(ch.finalPath, syn.error || 'parse error'), title: ch.finalPath, path: ch.finalPath }
      }
      await writeAppliedChanges(ctx.cwd, changes)
      const diff = combinedDiff(changes)
      const { added, removed } = diffStat(diff)
      const onePath = changes.length === 1 ? changes[0].finalPath : undefined
      return {
        ok: true,
        output: patchSummary(changes),
        title: changes.length === 1 ? onePath : 'Patch',
        path: onePath,
        diff,
        changedPaths: [...new Set(changes.flatMap((change) => [change.filePath, change.finalPath]).filter(Boolean))],
        meta: `${changes.length} file${changes.length === 1 ? '' : 's'} · +${added} −${removed}`,
      }
    }

    case 'search': {
      const query: string = args.query ?? ''
      const base = resolveWorkspacePath(ctx, args.path)
      if (isRemote(ctx)) {
        const backend = workspaceBackend(ctx)
        // `cwd` is consumed by a remote shell, so lexical containment is not enough:
        // an in-root path may be a symlink to another tenant or `/etc`. Require the SSH
        // host's canonical proof before grep starts walking it.
        if (!backend.assertContained) throw new Error('Remote backend cannot verify workspace containment.')
        await backend.assertContained(base)
        const excludes = [...SKIP_DIRS].map((d) => `--exclude-dir=${shArg(d)}`).join(' ')
        const cmd = `grep -rInE -- ${shArg(query)} ${excludes} . 2>/dev/null | head -n 100`
        const res = await backend.exec(cmd, { cwd: base })
        // Drop hits from credential files: grep dumps line contents, so without this the
        // remote path would surface in-repo .env/credentials that read_file blocks (the
        // local walk already filters these; mirror it here).
        const hits = res.stdout.split('\n').filter(Boolean)
          .map((l) => l.replace(/^\.\//, ''))
          .filter((l) => !blockedReadReason(l.split(':', 1)[0]))
        return { ok: true, output: hits.length ? hits.join('\n') : 'No matches.', title: `"${query}"`, meta: `${hits.length} matches` }
      }
      const hits = (await boundedRegexSearch(base, query, SKIP_DIRS))
        .map((line) => line.replace(/^\.\//, ''))
        .filter((line) => !blockedReadReason(line.split(':', 1)[0]))
        .slice(0, 100)
        .map((line) => {
          const match = /^([^:]+):(\d+):(.*)$/.exec(line)
          return match ? `${rel(ctx.cwd, path.join(base, match[1]))}:${match[2]}: ${match[3].trim().slice(0, 200)}` : line.slice(0, 240)
        })
      return { ok: true, output: hits.length ? hits.join('\n') : 'No matches.', title: `"${query}"`, meta: `${hits.length} matches` }
    }

    case 'glob': {
      const pattern = validateRelativePattern(String(args.pattern ?? ''))
      const base = resolveWorkspacePath(ctx, args.path)
      const rawMax = Number(args.max_results ?? DEFAULT_GLOB_RESULTS)
      const max = Math.max(1, Math.min(MAX_GLOB_RESULTS, Number.isFinite(rawMax) ? Math.floor(rawMax) : DEFAULT_GLOB_RESULTS))
      if (isRemote(ctx)) {
        const backend = workspaceBackend(ctx)
        // `find` has the same remote-cwd symlink boundary as grep above.
        if (!backend.assertContained) throw new Error('Remote backend cannot verify workspace containment.')
        await backend.assertContained(base)
        const namePart = pattern.split('/').pop() || pattern
        const cmd = `find . -type f -name ${shArg(namePart)} -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/dist/*' 2>/dev/null | sed 's|^\\./||' | head -n ${max}`
        const res = await backend.exec(cmd, { cwd: base })
        const remoteHits = res.stdout.split('\n').filter(Boolean).filter((p) => !blockedReadReason(p))
        return {
          ok: true,
          output: remoteHits.length ? remoteHits.join('\n') : 'No matches.',
          title: pattern,
          path: args.path ? relWorkspace(ctx, base) : undefined,
          meta: `${remoteHits.length}${remoteHits.length >= max ? '+' : ''} files`,
        }
      }
      const hits = await collectFiles(ctx.cwd, base, pattern, max)
      return {
        ok: true,
        output: hits.length ? hits.join('\n') : 'No matches.',
        title: pattern,
        path: args.path ? rel(ctx.cwd, base) : undefined,
        meta: `${hits.length}${hits.length >= max ? '+' : ''} files`,
      }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
