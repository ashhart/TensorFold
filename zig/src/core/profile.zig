//! Reviewed switch metadata and environment-first lookup; stable leaves every existing family default intact.
const std = @import("std");

pub const Profile = enum { stable };
pub const Class = enum { identical, arithmetic, experimental, debug };
pub const Switch = enum {
    TF_QWEN_KV_PAGES,
    TF_QWEN_KV_PAGES_ALL,
    TF_QWEN_KV_SHARE,
    TF_KV_SHARE,
    TF_FLASHNEXT_KV_PAGES,
    TF_FLASHNEXT_KV_PAGES_ALL,
    TF_KV_PAGES,
    TF_GLM_KV_HOST,
    TF_GLM_KV_HOST_SHARED,
    TF_GLM_PROMPT_TAIL,
    TENSORFOLD_KOLIBRI_KV_PAGES,
    TENSORFOLD_DORY_KV_PAGES,
    GEMMA_DRAFT_BF16,
    TF_ANE_COLUMNS,
    TF_PROMPT_ROWS,
    slide_learn_mode,
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
    families: []const []const u8,
    backend: []const u8 = "cuda",
    class: Class,
    effect: []const u8,
    evidence: []const u8,
    syntax: Syntax = .count,
    stable: ?[:0]const u8 = null,

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
};

const qwen: []const []const u8 = &.{ "qwen3_5", "qwen3_5_moe" };
const nemotron: []const []const u8 = &.{ "nemotron_h", "nemotron_h_omni" };
const slide_cuda: []const []const u8 = &.{ "nemotron_h", "nemotron_h_omni", "qwen3_5" };
const flash: []const []const u8 = &.{"qwen4_exp"};

pub const entries = [_]Entry{
    .{ .id = .TF_QWEN_KV_PAGES, .families = qwen, .class = .identical, .effect = "KV pool page size; memory and workload tradeoffs", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = qwen, .class = .debug, .syntax = .one, .effect = "Put the engine's own stream on pages too", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_SHARE, .families = qwen, .class = .identical, .effect = "Keep prompt pages and bounded recurrent snapshots", .evidence = "#1357" },
    .{ .id = .TF_KV_SHARE, .families = &.{ "gemma4", "gemma4_unified", "kolibri1", "spark2_5", "glm5_next" }, .class = .identical, .effect = "Keep prompt states over the core KV pool", .evidence = "#1657" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES, .families = flash, .class = .experimental, .effect = "KV pool page size; fast-profile qualification pending", .evidence = "#1660" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES_ALL, .families = flash, .class = .debug, .syntax = .one, .effect = "Put every stream on pool pages for checks", .evidence = "#1660" },
    .{ .id = .TF_KV_PAGES, .families = &.{"glm5_next"}, .class = .identical, .syntax = .auto, .effect = "GLM KV pool pages; auto from two ranks with BF16, windows with FP8", .evidence = "#1322" },
    .{ .id = .TF_KV_PAGES, .families = &.{ "nemotron_h", "nemotron_h_omni" }, .class = .identical, .report_values = &.{ "0", "all" }, .effect = "Nemotron KV pool pages by default; exact 0 disables, exact all pages every stream", .evidence = "#781" },
    .{ .id = .TF_GLM_KV_HOST, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep the DSA layers' latent rows on pinned host pages", .evidence = "#1795" },
    .{ .id = .TF_GLM_KV_HOST_SHARED, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "One copy of those host pages across the ranks on one host", .evidence = "#1795" },
    .{ .id = .TF_GLM_PROMPT_TAIL, .families = &.{"glm5_next"}, .class = .identical, .syntax = .one, .effect = "Keep a raw prompt's state at its last chunk start, from the chat states' budget", .evidence = "#2226" },
    .{ .id = .TENSORFOLD_KOLIBRI_KV_PAGES, .families = &.{"kolibri1"}, .class = .arithmetic, .report_values = &.{"0"}, .effect = "Exact 0 disables requested pool pages; other text keeps the option", .evidence = "#2370" },
    .{ .id = .TENSORFOLD_DORY_KV_PAGES, .families = &.{"dory"}, .class = .arithmetic, .syntax = .not_zero, .effect = "Present text except exact 0 selects pool pages instead of default windows", .evidence = "#2370" },
    .{ .id = .GEMMA_DRAFT_BF16, .families = &.{ "gemma4", "gemma4_unified" }, .backend = "metal", .class = .debug, .report_values = &.{ "all", "fc", "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj" }, .effect = "Keep selected drafter stacks in stored BF16 instead of quantizing them", .evidence = "#2370" },
    .{ .id = .TF_ANE_COLUMNS, .families = &.{"qwen3_5"}, .backend = "metal", .class = .experimental, .effect = "FP16 Neural Engine prompt columns; exact output not qualified", .evidence = "#2370" },
    .{ .id = .TF_PROMPT_ROWS, .families = qwen, .backend = "metal", .class = .arithmetic, .syntax = .present, .effect = "Explicit prompt frame capacity and kept-state key geometry", .evidence = "#2370" },
    .{ .id = .slide_learn_mode, .families = slide_cuda, .backend = "multiple", .scopes = &.{ .{ .backend = "cuda", .families = slide_cuda }, .{ .backend = "metal", .families = nemotron } }, .members = &.{ .{ .name = "TF_SLIDE_POCKET_KEY", .syntax = .auto }, .{ .name = "TF_SLIDE_EDIT", .syntax = .present, .families = nemotron }, .{ .name = "TF_SLIDE_PROBE", .syntax = .present, .families = nemotron }, .{ .name = "TF_SLIDE_ROUTED", .syntax = .present, .families = nemotron } }, .class = .experimental, .effect = "Pocket keying and presence-selected edit, probe or routed learning modes", .evidence = "#2370" },
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
