//! Reviewed switch metadata and environment-first lookup; stable leaves every existing family default intact.
const std = @import("std");

pub const Profile = enum { stable };
pub const Class = enum { identical, arithmetic, experimental, debug };
pub const Switch = enum {
    TF_QWEN_KV_PAGES,
    TF_QWEN_KV_PAGES_ALL,
    TF_QWEN_KV_SHARE,
    TF_GLM_KV_SHARE,
    TF_KV_SHARE,
    TF_FLASHNEXT_KV_PAGES,
    TF_FLASHNEXT_KV_PAGES_ALL,
    TF_KV_PAGES,
};
pub const Syntax = enum { count, one, auto };
pub const Entry = struct {
    id: Switch,
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
        if (!std.mem.eql(u8, e.backend, backend)) return false;
        for (e.families) |f| if (std.mem.eql(u8, f, family)) return true;
        return false;
    }

    pub fn differs(e: Entry, raw: []const u8) bool {
        return switch (e.syntax) {
            .one => std.mem.eql(u8, raw, "1"),
            .count => (std.fmt.parseInt(usize, raw, 10) catch return true) != 0,
            .auto => !std.mem.eql(u8, raw, "auto"),
        };
    }
};

const qwen: []const []const u8 = &.{ "qwen3_5", "qwen3_5_moe" };
const flash: []const []const u8 = &.{"qwen4_exp"};

pub const entries = [_]Entry{
    .{ .id = .TF_QWEN_KV_PAGES, .families = qwen, .class = .identical, .effect = "KV pool page size; memory and workload tradeoffs", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = qwen, .class = .debug, .syntax = .one, .effect = "Put the engine's own stream on pages too", .evidence = "#1357" },
    .{ .id = .TF_QWEN_KV_SHARE, .families = qwen, .class = .identical, .effect = "Keep prompt pages and bounded recurrent snapshots", .evidence = "#1357" },
    .{ .id = .TF_GLM_KV_SHARE, .families = &.{"glm5_next"}, .class = .identical, .effect = "Keep prompt states on pool pages; requires TF_KV_PAGES", .evidence = "#1510" },
    .{ .id = .TF_KV_SHARE, .families = &.{ "gemma4", "gemma4_unified", "kolibri1", "spark2_5" }, .class = .identical, .effect = "Keep prompt states over the core KV pool", .evidence = "#1657" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES, .families = flash, .class = .experimental, .effect = "KV pool page size; fast-profile qualification pending", .evidence = "#1660" },
    .{ .id = .TF_FLASHNEXT_KV_PAGES_ALL, .families = flash, .class = .debug, .syntax = .one, .effect = "Put every stream on pool pages for checks", .evidence = "#1660" },
    .{ .id = .TF_KV_PAGES, .families = &.{"glm5_next"}, .class = .identical, .syntax = .auto, .effect = "GLM KV pool pages; auto from two ranks with BF16, windows with FP8", .evidence = "#1322" },
};

comptime {
    if (entries.len != @typeInfo(Switch).@"enum".field_names.len) @compileError("Every profile switch needs one registry entry");
    for (entries, 0..) |e, i| {
        if (@backingInt(e.id) != i) @compileError("Profile entries must follow Switch order");
        if (e.stable != null) @compileError("The stable foundation must not supply new switch defaults");
    }
}

pub fn entry(id: Switch) Entry {
    return entries[@backingInt(id)];
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
    const raw = std.c.getenv(entry(id).name());
    return resolve(.stable, id, if (raw) |v| std.mem.span(v) else null);
}

test {
    _ = @import("profile_test.zig");
}
