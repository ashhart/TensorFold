// Standalone entry: run TensorFold's tools as an MCP server over stdio, so Codex/Claude
// (CLI or app) and any MCP client can use them. Register, e.g.:
//   codex mcp add vontra -- node <repo>/node_modules/.bin/jiti <repo>/electron/agent/mcp-server-main.ts <cwd>
//   claude mcp add vontra -- node .../jiti .../mcp-server-main.ts <cwd>
// The working directory is argv[2] or $VONTRA_MCP_CWD (default: process.cwd()).
//
// Only READ-ONLY tools that execute with a {cwd}-only context are exposed here (file read,
// search, code-intelligence, web). Mutating tools (write_file, edit_file, apply_patch,
// run_command) are intentionally omitted because the standalone server has no access to
// TensorFold's approval gate (requestApproval / risk-band / saved-policy machinery) — exposing
// them would let a connected MCP client write files or run commands with no human approval.
// Agentic tools that need the live TensorFold app (sub-agents, connectors, browser/controller,
// MCP-proxied) are also omitted.

import { runMcpServerStdio, vontraToolsToMcp } from './mcp-server'
import { buildToolSchemas } from './tool-schema-builder'
import { executeTool } from './tools'

// Read-only subset only — see header comment for rationale.
export const STANDALONE_MCP_TOOLS = new Set([
  'list_dir', 'read_file', 'search', 'glob',
  'ast_grep_search', 'ast_grep_rewrite_preview', 'semgrep_scan', 'safe_refactor_preview',
  'codebase_map', 'symbol_index', 'retrieval_search', 'repo_wiki', 'web_search', 'web_fetch',
])

export function standaloneMcpTools() {
  const schemas = buildToolSchemas().filter((s: any) => STANDALONE_MCP_TOOLS.has(s?.function?.name))
  return vontraToolsToMcp(schemas)
}

function main() {
  const cwd = process.argv[2] || process.env.VONTRA_MCP_CWD || process.cwd()
  runMcpServerStdio({
    tools: standaloneMcpTools(),
    serverInfo: { name: 'TensorFold', version: '0.1.0' },
    callTool: async (name, args) => {
      if (!STANDALONE_MCP_TOOLS.has(name)) {
        return { ok: false, output: `Tool "${name}" is not available over the standalone TensorFold MCP server.` }
      }
      const result = await executeTool(name, args, { cwd })
      return { ok: result.ok, output: result.output }
    },
  })
}

// Run only when invoked directly (not when imported by a test).
if (process.argv[1] && /mcp-server-main\.(ts|js|mjs)$/.test(process.argv[1])) main()
