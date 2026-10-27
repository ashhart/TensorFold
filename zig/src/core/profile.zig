//! Reviewed switch metadata and environment-first lookup; stable leaves every existing family default intact.
pub const startup_receipt = @import("startup_receipt.zig");
const std = @import("std");

pub const Profile = enum { stable };
pub const Class = enum { identical, arithmetic, experimental, debug };
pub const Switch = enum {
    TF_QWEN_KV_PAGES,
    TF_QWEN_KV_PAGES_ALL,
    TF_QWEN_KV_SHARE,
    TF_QWEN_REQUEST_STARTS,
    TF_GLM_REQUEST_STARTS,
    TF_KV_SHARE,
    TF_FLASHNEXT_KV_PAGES,
    TF_FLASHNEXT_KV_PAGES_ALL,
    TF_KV_PAGES,
    TF_GLM_KV_HOST,
    TF_GLM_KV_HOST_SHARED,
    TF_GLM_PROMPT_TAIL,
    TF_GLM_HOST_CACHE,
    TF_GLM_VISION_HOST,
    TF_GLM_PDL,
    TF_GLM_PROMPT_STEPS,
    TF_GLM_SHARED,
    TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS,
    TENSORFOLD_KOLIBRI_KV_PAGES,
    TENSORFOLD_DORY_KV_PAGES,
    GEMMA_DRAFT_BF16,
    TF_ANE_COLUMNS,
    TF_PROMPT_ROWS,
    TENSORFOLD_GEMMA4_KV_PAGES,
    TENSORFOLD_SPARK25_KV_PAGES,
    TENSORFOLD_LFM2_KV_PAGES,
    TF_NEMOTRON_METAL_KV_PAGES,
    slide_learn_mode,
    TF_DSV41_EXPERT_TOPP,
    TF_DSV41_NUCLEUS,
    TF_DSV41_CONTEXT,
    TF_DSV41_LAYERS,
    TF_DSV41_OWN_PREFILL,
    TF_DSV41_PREFILL_PIECES,
    TF_DSV41_SLOTS,
    TF_DSV41_POOL_TOKENS,
    TF_DSV41_KV_SPLIT,
    TF_DSV41_KV_SPLIT_UNION_MIB,
    TF_DSV41_NEEDS_POOL,
    TF_DSV41_MHC_DEFER,
    TF_DSV41_DRAFTS,
    TF_DSV41_SAMPLING,
    TF_DSV41_IMAGES,
    TF_DSV41_BIAS_VL,
    TF_DSV41_ENGRAM_DIR,
    TF_DSV41_SCHEMA_PROMPT,
    TF_DSV41_ROWS_CAP,
    TF_DSV41_TOOL_GRAMMAR,
};
pub const Syntax = enum { count, one, auto, present, not_zero };
pub const Scope = struct { backend: []const u8, families: []const []const u8 };
pub const Member = struct {
    name: [:0]const u8,
    syntax: Syntax,
    families: ?[]const []const u8 = null,
    pub fn applies(m: Member, family: []const u8) bool {
        for (m.families orelse return true) |f| if (std.mem.eql(u8, f, family)) return true;
        return false;
    }
    pub fn differs(m: Member, raw: []const u8) bool {
        return different(m.syntax, raw);
    }
};

fn different(syntax: Syntax, raw: []const u8) bool {
    return switch (syntax) {
        .one => std.mem.eql(u8, raw, "1"),
        .count => (std.fmt.parseInt(usize, raw, 10) catch return true) != 0,
        .auto => !std.mem.eql(u8, raw, "auto"),
        .present => true,
        .not_zero => !std.mem.eql(u8, raw, "0"),
    };
}
pub const Entry = struct {
    id: Switch,
    /// A grouped row uses real member names for lookup, never its synthetic id.
    members: ?[]const Member = null,
    /// A group spanning more than one backend publishes its exact scopes here.
    scopes: ?[]const Scope = null,
    /// When present, these exact values alone differ from the family default; otherwise use syntax.
    report_values: ?[]const []const u8 = null,
    redact: bool = false,
    families: []const []const u8,
    backend: []const u8 = "cuda",
    class: Class,
    effect: []const u8,
    evidence: []const u8,
    syntax: Syntax = .count,
    stable: ?[:0]const u8 = null,
    /// Unset, the row counts as on ("1") while this switch is on: a family that turns it on by another name.
    on_with: ?Switch = null,

    pub fn name(e: Entry) [:0]const u8 {
        return @tagName(e.id);
    }

    pub fn applies(e: Entry, family: []const u8, backend: []const u8) bool {
        if (e.scopes) |scopes| {
            for (scopes) |scope| {
                if (!std.mem.eql(u8, scope.backend, backend)) continue;
                for (scope.families) |f| if (std.mem.eql(u8, f, family)) return true;
            }
            return false;
        }
        if (!std.mem.eql(u8, e.backend, backend)) return false;
        for (e.families) |f| if (std.mem.eql(u8, f, family)) return true;
        return false;
    }

    pub fn differs(e: Entry, raw: []const u8) bool {
        if (e.report_values) |values| {
            for (values) |v| if (std.mem.eql(u8, raw, v)) return true;
            return false;
        }
        return different(e.syntax, raw);
    }

    /// The row's value as served: its own when present, else "1" when `on_with`'s value (`by`) turns it on.
    pub fn served(e: Entry, own: ?[]const u8, by: ?[]const u8) ?[]const u8 {
        if (own != null or e.on_with == null) return own;
        const v = by orelse return null;
        return if (!blank(v) and different(entry(e.on_with.?).syntax, v)) "1" else null;
    }
};

const qwen: []const []const u8 = &.{ "qwen3_5", "qwen3_5_moe" };
const nemotron: []const []const u8 = &.{ "nemotron_h", "nemotron_h_omni" };
const slide_cuda: []const []const u8 = &.{ "nemotron_h", "nemotron_h_omni", "qwen3_5" };
const flash: []const []const u8 = &.{"qwen4_exp"};
/// The CUDA families whose host prompt cache the generic CUDA host builds (cuda_keep_others_test.zig checks the list).
pub const cuda_cache: []const []const u8 = &.{ "nemotron_h", "nemotron_h_omni", "glm5_next", "minimax_m3_vl", "qwen3_5", "qwen3_5_moe", "qwen4_exp", "kolibri1", "gemma4", "gemma4_unified", "spark2_5", "dory", "lfm2_moe" };

pub const entries = [_]Entry{
    .{ .id = .TF_QWEN_REQUEST_STARTS, .families = qwen, .class = .arithmetic, .syntax = .one, .effect = "Opt into CUDA message-start prompt cuts", .evidence = "#1334; default-off; GB10 six-turn: resumed equals fresh, one reply differs from unset" },
    .{ .id = .TF_GLM_REQUEST_STARTS, .families = &.{"glm5_next"}, .class = .arithmetic, .syntax = .one, .effect = "Opt into CUDA message-start prompt cuts", .evidence = "#2396; default-off; GB10 TP=2 receipts in #2404" },
    .{ .id = .TF_QWEN_KV_PAGES, .families = qwen, .class = .identical, .effect = "KV pool page size; memory and workload tradeoffs", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = qwen, .class = .debug, .syntax = .one, .effect = "Put the engine's own stream on pages too", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_SHARE, .families = qwen, .class = .identical, .effect = "Keep prompt pages and bounded recurrent snapshots", .evidence = "#1357" },
    .{ .id = .TF_KV_SHARE, .families = &.{ "gemma4", "gemma4_unified", "kolibri1", "spark2_5", "glm5_next", "dory" }, .class = .identical, .effect = "Keep prompt states over the core KV pool", .evidence = "#1657" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES, .families = flash, .class = .experimental, .effect = "KV pool page size; no profile qualifies it yet", .evidence = "#1660" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES_ALL, .families = flash, .class = .debug, .syntax = .one, .effect = "Put every stream on pool pages for checks", .evidence = "#1660" },
    .{ .id = .TF_KV_PAGES, .families = &.{"glm5_next"}, .class = .identical, .syntax = .auto, .effect = "GLM KV pool pages; auto from two ranks with BF16, windows with FP8", .evidence = "#1322" },
    .{ .id = .TF_KV_PAGES, .families = &.{ "nemotron_h", "nemotron_h_omni" }, .class = .identical, .report_values = &.{ "0", "all" }, .effect = "Nemotron KV pool pages by default; exact 0 disables, exact all pages every stream", .evidence = "#781" },
    .{ .id = .TF_GLM_KV_HOST, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep the DSA layers' latent rows on pinned host pages", .evidence = "#1795" },
    .{ .id = .TF_GLM_KV_HOST_SHARED, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "One copy of those host pages across the ranks on one host", .evidence = "#1795" },
    .{ .id = .TF_GLM_PROMPT_TAIL, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep a raw prompt's state at its last chunk start, from the chat states' budget", .evidence = "#2226" },
    .{ .id = .TF_GLM_HOST_CACHE, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep kept prompt states in pinned host memory instead of on the device", .evidence = "#2370" },
    .{ .id = .TF_GLM_VISION_HOST, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep the vision tower's weights in pinned host memory, staged a unit at a time", .evidence = "#2370" },
    .{ .id = .TF_GLM_PDL, .families = &.{"glm5_next"}, .class = .identical, .report_values = &.{"0"}, .effect = "Exact 0 turns off the matmuls' dependent launches", .evidence = "#2370" },
    .{ .id = .TF_GLM_PROMPT_STEPS, .families = &.{"glm5_next"}, .class = .identical, .report_values = &.{"0"}, .effect = "Exact 0 runs rank 0's prompt passes whole, not in chunks", .evidence = "#2370" },
    .{ .id = .TF_GLM_SHARED, .families = &.{"glm5_next"}, .class = .identical, .report_values = &.{"0"}, .effect = "Exact 0 runs each of rank 0's windows in its own forward", .evidence = "#2370" },
    .{ .id = .TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS, .families = cuda_cache, .class = .identical, .syntax = .one, .effect = "A new prompt state that fits only by evicting another conversation's is not kept", .evidence = "#2346" },
    .{ .id = .TENSORFOLD_KOLIBRI_KV_PAGES, .families = &.{"kolibri1"}, .class = .arithmetic, .report_values = &.{"0"}, .effect = "Exact 0 disables requested pool pages; other text keeps the option", .evidence = "#2370" },
    .{ .id = .TENSORFOLD_DORY_KV_PAGES, .families = &.{"dory"}, .class = .arithmetic, .syntax = .not_zero, .on_with = .TF_KV_SHARE, .effect = "Present text except exact 0 selects pool pages instead of default windows", .evidence = "#2370" },
    .{ .id = .GEMMA_DRAFT_BF16, .families = &.{ "gemma4", "gemma4_unified" }, .backend = "metal", .class = .debug, .report_values = &.{ "all", "fc", "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj" }, .effect = "Keep selected drafter stacks in stored BF16 instead of quantizing them", .evidence = "#2370" },
    .{ .id = .TF_ANE_COLUMNS, .families = &.{"qwen3_5"}, .backend = "metal", .class = .experimental, .effect = "FP16 Neural Engine prompt columns; exact output not qualified", .evidence = "#2370" },
    .{ .id = .TF_PROMPT_ROWS, .families = qwen, .backend = "metal", .class = .arithmetic, .syntax = .present, .effect = "Explicit prompt frame capacity and kept-state key geometry", .evidence = "#2370" },
    .{ .id = .slide_learn_mode, .families = slide_cuda, .backend = "multiple", .scopes = &.{ .{ .backend = "cuda", .families = slide_cuda }, .{ .backend = "metal", .families = nemotron } }, .members = &.{ .{ .name = "TF_SLIDE_POCKET_KEY", .syntax = .auto }, .{ .name = "TF_SLIDE_EDIT", .syntax = .present, .families = nemotron }, .{ .name = "TF_SLIDE_PROBE", .syntax = .present, .families = nemotron }, .{ .name = "TF_SLIDE_ROUTED", .syntax = .present, .families = nemotron } }, .class = .experimental, .effect = "Pocket keying and presence-selected edit, probe or routed learning modes", .evidence = "#2370" },
    .{ .id = .TENSORFOLD_GEMMA4_KV_PAGES, .families = &.{ "gemma4", "gemma4_unified" }, .class = .identical, .report_values = &.{"0"}, .effect = "Global caches lease pool pages by default; 0 keeps them linear", .evidence = "#1619" },
    .{ .id = .TENSORFOLD_SPARK25_KV_PAGES, .families = &.{"spark2_5"}, .class = .identical, .report_values = &.{"0"}, .effect = "Full caches lease pool pages by default; 0 keeps them linear", .evidence = "#1653" },
    .{ .id = .TENSORFOLD_LFM2_KV_PAGES, .families = &.{"lfm2_moe"}, .class = .identical, .report_values = &.{"0"}, .effect = "Attention caches lease pool pages by default; 0 keeps fixed pages a stream", .evidence = "#1503" },
    .{ .id = .TF_NEMOTRON_METAL_KV_PAGES, .families = &.{"nemotron_h"}, .backend = "metal", .class = .identical, .syntax = .one, .effect = "Only exact 1 puts keys and values in the core pool's pages; unset, whole caches made at open", .evidence = "#1537" },
    .{ .id = .TF_DSV41_EXPERT_TOPP, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Expert top-p routing mass", .evidence = "#2370" },
    .{ .id = .TF_DSV41_NUCLEUS, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Nucleus candidate count", .evidence = "#2370" },
    .{ .id = .TF_DSV41_CONTEXT, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Slot position limit", .evidence = "#2370" },
    .{ .id = .TF_DSV41_LAYERS, .families = &.{"deepseek_v41"}, .class = .debug, .syntax = .present, .effect = "Loaded backbone layer subset", .evidence = "#2370" },
    .{ .id = .TF_DSV41_OWN_PREFILL, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Prompt prefill program; ranks must agree", .evidence = "#2370" },
    .{ .id = .TF_DSV41_PREFILL_PIECES, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Host prompt-piece scheduler", .evidence = "#2370" },
    .{ .id = .TF_DSV41_SLOTS, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Live serving slots", .evidence = "#2370" },
    .{ .id = .TF_DSV41_POOL_TOKENS, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Paged pool token capacity", .evidence = "#2370" },
    .{ .id = .TF_DSV41_KV_SPLIT, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Split ownership of pool rows", .evidence = "#2370" },
    .{ .id = .TF_DSV41_KV_SPLIT_UNION_MIB, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Split pool union capacity", .evidence = "#2370" },
    .{ .id = .TF_DSV41_NEEDS_POOL, .families = &.{"deepseek_v41"}, .class = .debug, .syntax = .present, .effect = "Force pool geometry in the AOT asset planner only", .evidence = "#2370" },
    .{ .id = .TF_DSV41_MHC_DEFER, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Deferred mHC coefficient launches", .evidence = "#2370" },
    .{ .id = .TF_DSV41_DRAFTS, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "DSpark drafting policy", .evidence = "#2370" },
    .{ .id = .TF_DSV41_SAMPLING, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Device sampler", .evidence = "#2370" },
    .{ .id = .TF_DSV41_IMAGES, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Native image segments in the AOT plan", .evidence = "#2370" },
    .{ .id = .TF_DSV41_BIAS_VL, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Vision-bias calls in the AOT plan", .evidence = "#2370" },
    .{ .id = .TF_DSV41_ENGRAM_DIR, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .redact = true, .effect = "Real Engram rows instead of synthetic rows", .evidence = "#2370" },
    .{ .id = .TF_DSV41_SCHEMA_PROMPT, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Response schema in the encoded prompt", .evidence = "#2370" },
    .{ .id = .TF_DSV41_ROWS_CAP, .families = &.{"deepseek_v41"}, .class = .arithmetic, .syntax = .present, .effect = "Rows in each forward bucket", .evidence = "#2370" },
    .{ .id = .TF_DSV41_TOOL_GRAMMAR, .families = &.{"deepseek_v41"}, .class = .experimental, .syntax = .present, .effect = "Legacy alias of the server-wide auto-tool grammar policy", .evidence = "#2370" },
};

comptime {
    @setEvalBranchQuota(20000);
    for (std.enums.values(Switch)) |id| {
        var found = false;
        for (entries) |e| if (e.id == id) {
            found = true;
        };
        if (!found) @compileError("Every profile switch needs a registry entry");
    }
    for (entries, 0..) |e, i| {
        if (e.stable != null) @compileError("The stable foundation must not supply new switch defaults");
        for (entries[0..i]) |previous| {
            if (previous.id != e.id) continue;
            for (e.families) |family| for ([_][]const u8{ "cuda", "metal", "hip", "xpu" }) |backend| {
                if (e.applies(family, backend) and previous.applies(family, backend))
                    @compileError("Profile switch rows must have disjoint family/backend scopes");
            };
        }
    }
}

/// Rows for one variable share its lookup and stable default; reporting uses each row's family/backend scope.
pub fn entry(id: Switch) Entry {
    for (entries) |e| if (e.id == id) return e;
    unreachable;
}

/// A present environment value wins byte for byte, even empty, zero, or invalid for the family's parser.
pub fn resolve(profile: Profile, id: Switch, environment: ?[:0]const u8) ?[:0]const u8 {
    if (environment) |v| return v;
    return switch (profile) {
        .stable => entry(id).stable,
    };
}

/// The only served profile is stable; no global setting or environment mutation is needed to select it.
pub inline fn get(comptime id: Switch) ?[:0]const u8 {
    return value(.stable, id);
}

/// `get` for a reader that parses the text: empty or whitespace-only text reads as unset (`docs/environment.md`).
pub inline fn given(comptime id: Switch) ?[:0]const u8 {
    const v = get(id) orelse return null;
    return if (blank(v)) resolve(.stable, id, null) else v;
}

/// Empty or whitespace-only text, which a parsing reader takes as unset.
fn blank(v: []const u8) bool {
    return std.mem.trim(u8, v, " \t\r\n").len == 0;
}

/// One switch's value for a profile chosen at runtime: the same environment-first rule as `get`.
pub fn value(profile: Profile, id: Switch) ?[:0]const u8 {
    if (entry(id).members != null) return null;
    const raw = std.c.getenv(entry(id).name());
    return resolve(profile, id, if (raw) |v| std.mem.span(v) else null);
}

/// The registry row a switch's environment name belongs to; null for a name the table does not cover.
pub fn registered(name: []const u8) ?Switch {
    inline for (@typeInfo(Switch).@"enum".field_names, @typeInfo(Switch).@"enum".field_values) |n, v| {
        if (!std.mem.eql(u8, n, "slide_learn_mode") and std.mem.eql(u8, name, n)) return @fromBackingInt(@intCast(v));
    }
    for (entry(.slide_learn_mode).members.?) |member|
        if (std.mem.eql(u8, name, member.name)) return .slide_learn_mode;
    return null;
}

/// A runtime name lookup: a registered switch resolves through the profile, any other name through libc as before.
pub fn lookup(name: [:0]const u8) ?[:0]const u8 {
    const raw = if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    if (registered(name)) |id| return resolve(.stable, id, raw);
    return raw;
}

test {
    _ = @import("profile_test.zig");
}
