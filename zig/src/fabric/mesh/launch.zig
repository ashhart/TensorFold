//! One launch description per rank: the server's flags and a family's legacy rank variables resolved once.
const std = @import("std");
const mesh_mod = @import("mesh.zig");

/// The families whose rank variables are aliases of the flags; `other` has none (Flash Next, HIP, ...).
pub const Family = enum { glm, minimax, nemotron, other };

/// The launch flags as given: null when absent, so an alias can fill the field.
pub const Flags = struct {
    tp: ?u32 = null,
    pp: ?u32 = null,
    rank: ?u32 = null,
    master: ?[]const u8 = null,
    port: ?u16 = null,
};

/// Where the world and rank came from, for the start-up line.
pub const Source = enum { flags, alias, mixed, default };

pub const Launch = struct {
    tensor: u32,
    pipeline: u32,
    rank: u32,
    master: ?[]const u8,
    port: u16,
    from: Source,
    /// The alias names read (null when none), for the server's warning and retirement check.
    world_name: ?[:0]const u8 = null,
    rank_name: ?[:0]const u8 = null,

    pub fn mesh(l: Launch) mesh_mod.Error!mesh_mod.Mesh {
        return mesh_mod.Mesh.init(.{ .tensor = l.tensor, .pipeline = l.pipeline }, l.rank);
    }
};

pub const Error = error{ PartialAlias, BadAlias, BadFlag, Disagree, NoPlace, PipelineNotServed };

/// `--master-port`'s default (core/engine_api.zig).
pub const default_port: u16 = 29551;
/// The most ranks a launch names (the fabric/tp bootstrap's bound).
pub const max_ranks: u32 = 64;

pub const Names = struct { world: [:0]const u8, rank: [:0]const u8 };

/// A family's legacy world and rank variables.
pub fn names(f: Family) ?Names {
    return switch (f) {
        .glm => .{ .world = "TF_GLM_WORLD", .rank = "TF_GLM_RANK" },
        .minimax => .{ .world = "TF_MINIMAX_WORLD", .rank = "TF_MINIMAX_RANK" },
        .nemotron => .{ .world = "TF_NEMOTRON_WORLD", .rank = "TF_NEMOTRON_RANK" },
        .other => null,
    };
}

/// Flags and aliases merged field by field (`lookup.get(name) ?[]const u8`); bounds the family checks stay its own.
pub fn resolve(f: Flags, lookup: anytype, family: Family) Error!Launch {
    // 1. the alias pair: both names or neither, each a whole number, as the family readers took them
    var alias_world: ?u32 = null;
    var alias_rank: ?u32 = null;
    var read: ?Names = null;
    if (names(family)) |n| {
        const w = lookup.get(n.world);
        const r = lookup.get(n.rank);
        if ((w == null) != (r == null)) return error.PartialAlias;
        if (w != null) {
            alias_world = std.fmt.parseInt(u32, w.?, 10) catch return error.BadAlias;
            alias_rank = std.fmt.parseInt(u32, r.?, 10) catch return error.BadAlias;
            read = n;
        }
    }
    // 2. the flags' own ranges
    if (f.tp) |t| if (t < 1 or t > max_ranks) return error.BadFlag;
    if (f.pp) |p| {
        if (p < 1 or p > max_ranks) return error.BadFlag;
        if (p > 1) return error.PipelineNotServed;
    }
    if (f.tp != null and f.rank != null and f.rank.? >= f.tp.?) return error.BadFlag;
    // 3. merge: a field given by both must agree
    const world = try merged(f.tp, alias_world);
    const rank = try merged(f.rank, alias_rank);
    // 4. defaults: GLM never had one (the family refuses with its own message)
    if (world == null and family == .glm) return error.NoPlace;
    const out: Launch = .{
        .tensor = world orelse 1,
        .pipeline = 1,
        .rank = rank orelse 0,
        .master = f.master,
        .port = f.port orelse default_port,
        .from = source(f.tp != null or f.rank != null, read != null),
        .world_name = if (read) |n| n.world else null,
        .rank_name = if (read) |n| n.rank else null,
    };
    // 5. bounds where a flag gave a value; alias-only values reach the family's own check unchanged
    if ((f.tp != null or f.rank != null) and out.rank >= out.tensor) return error.BadFlag;
    return out;
}

fn merged(flag: ?u32, alias: ?u32) Error!?u32 {
    if (flag != null and alias != null and flag.? != alias.?) return error.Disagree;
    return flag orelse alias;
}

fn source(flags: bool, alias: bool) Source {
    if (flags and alias) return .mixed;
    if (flags) return .flags;
    if (alias) return .alias;
    return .default;
}

const testing = std.testing;

const Map = struct {
    pairs: []const [2][]const u8,
    pub fn get(m: Map, name: []const u8) ?[]const u8 {
        for (m.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

fn env(pairs: []const [2][]const u8) Map {
    return .{ .pairs = pairs };
}

test "flags alone, aliases alone and neither give today's places" {
    const none = env(&.{});
    const flags = try resolve(.{ .tp = 2, .rank = 1, .master = "10.0.0.1" }, none, .glm);
    try testing.expectEqual(@as(u32, 2), flags.tensor);
    try testing.expectEqual(@as(u32, 1), flags.rank);
    try testing.expectEqual(Source.flags, flags.from);
    try testing.expectEqual(default_port, flags.port);
    try testing.expectEqualStrings("10.0.0.1", flags.master.?);
    const alias = try resolve(.{}, env(&.{ .{ "TF_GLM_WORLD", "3" }, .{ "TF_GLM_RANK", "2" } }), .glm);
    try testing.expectEqual(@as(u32, 3), alias.tensor);
    try testing.expectEqual(@as(u32, 2), alias.rank);
    try testing.expectEqual(Source.alias, alias.from);
    try testing.expectEqualStrings("TF_GLM_WORLD", alias.world_name.?);
    try testing.expectError(error.NoPlace, resolve(.{}, none, .glm));
    for ([_]Family{ .minimax, .nemotron, .other }) |fam| {
        const one = try resolve(.{}, none, fam);
        try testing.expectEqual(@as(u32, 1), one.tensor);
        try testing.expectEqual(@as(u32, 0), one.rank);
        try testing.expectEqual(Source.default, one.from);
    }
}

test "aliases are read only for their own family" {
    const glm_env = env(&.{ .{ "TF_GLM_WORLD", "2" }, .{ "TF_GLM_RANK", "1" } });
    try testing.expectEqual(@as(u32, 1), (try resolve(.{}, glm_env, .minimax)).tensor);
    try testing.expectEqual(@as(u32, 1), (try resolve(.{}, glm_env, .other)).tensor);
    const mm = try resolve(.{}, env(&.{ .{ "TF_MINIMAX_WORLD", "2" }, .{ "TF_MINIMAX_RANK", "1" } }), .minimax);
    try testing.expectEqual(@as(u32, 2), mm.tensor);
    const nm = try resolve(.{}, env(&.{ .{ "TF_NEMOTRON_WORLD", "2" }, .{ "TF_NEMOTRON_RANK", "0" } }), .nemotron);
    try testing.expectEqual(@as(u32, 2), nm.tensor);
}

test "a partial or malformed pair refuses, as the family readers did" {
    for ([_]Family{ .glm, .minimax, .nemotron }) |fam| {
        const n = names(fam).?;
        try testing.expectError(error.PartialAlias, resolve(.{}, env(&.{.{ n.world, "2" }}), fam));
        try testing.expectError(error.PartialAlias, resolve(.{}, env(&.{.{ n.rank, "0" }}), fam));
        try testing.expectError(error.BadAlias, resolve(.{}, env(&.{ .{ n.world, "two" }, .{ n.rank, "0" } }), fam));
        try testing.expectError(error.BadAlias, resolve(.{}, env(&.{ .{ n.world, "2" }, .{ n.rank, "-1" } }), fam));
    }
}

test "alias text parses as the family readers' parseInt did, a u32 in place of their usize" {
    for ([_]Family{ .glm, .minimax, .nemotron }) |fam| {
        const n = names(fam).?;
        try testing.expectEqual(@as(u32, 2), (try resolve(.{}, env(&.{ .{ n.world, "+2" }, .{ n.rank, "0" } }), fam)).tensor);
        try testing.expectError(error.BadAlias, resolve(.{}, env(&.{ .{ n.world, " 2" }, .{ n.rank, "0" } }), fam));
        try testing.expectError(error.BadAlias, resolve(.{}, env(&.{ .{ n.world, "" }, .{ n.rank, "0" } }), fam));
        // above u32: today's usize readers parsed it, then refused it as too many ranks; it refuses here instead
        try testing.expectError(error.BadAlias, resolve(.{}, env(&.{ .{ n.world, "4294967296" }, .{ n.rank, "0" } }), fam));
        const merged_rank = try resolve(.{ .rank = 1 }, env(&.{ .{ n.world, "2" }, .{ n.rank, "1" } }), fam);
        try testing.expectEqual(Source.mixed, merged_rank.from);
        try testing.expectError(error.Disagree, resolve(.{ .tp = 4 }, env(&.{ .{ n.world, "2" }, .{ n.rank, "1" } }), fam));
    }
}

test "alias-only values out of bounds pass through to the family's own check" {
    const l = try resolve(.{}, env(&.{ .{ "TF_GLM_WORLD", "2" }, .{ "TF_GLM_RANK", "2" } }), .glm);
    try testing.expectEqual(@as(u32, 2), l.rank);
    const z = try resolve(.{}, env(&.{ .{ "TF_GLM_WORLD", "0" }, .{ "TF_GLM_RANK", "0" } }), .glm);
    try testing.expectEqual(@as(u32, 0), z.tensor);
}

test "flags and aliases merge field by field and must agree" {
    const pair = env(&.{ .{ "TF_GLM_WORLD", "3" }, .{ "TF_GLM_RANK", "2" } });
    const rank_only = try resolve(.{ .rank = 2 }, pair, .glm);
    try testing.expectEqual(@as(u32, 3), rank_only.tensor);
    try testing.expectEqual(Source.mixed, rank_only.from);
    const master_only = try resolve(.{ .master = "h", .port = 7 }, pair, .glm);
    try testing.expectEqual(@as(u32, 3), master_only.tensor);
    try testing.expectEqual(@as(u16, 7), master_only.port);
    try testing.expectEqual(Source.alias, master_only.from);
    try testing.expectEqual(@as(u32, 3), (try resolve(.{ .tp = 3, .rank = 2 }, pair, .glm)).tensor);
    try testing.expectError(error.Disagree, resolve(.{ .tp = 2 }, pair, .glm));
    try testing.expectError(error.Disagree, resolve(.{ .rank = 1 }, pair, .glm));
}

test "flag bounds: tp 1 to 64, rank below tp, pp 1 until T8" {
    const none = env(&.{});
    try testing.expectEqual(@as(u32, 64), (try resolve(.{ .tp = 64, .rank = 63 }, none, .other)).tensor);
    try testing.expectError(error.BadFlag, resolve(.{ .tp = 0 }, none, .other));
    try testing.expectError(error.BadFlag, resolve(.{ .tp = 65 }, none, .other));
    try testing.expectError(error.BadFlag, resolve(.{ .tp = 2, .rank = 2 }, none, .other));
    try testing.expectError(error.BadFlag, resolve(.{ .rank = 1 }, none, .other));
    try testing.expectError(error.PipelineNotServed, resolve(.{ .pp = 2 }, none, .other));
    try testing.expectError(error.BadFlag, resolve(.{ .pp = 0 }, none, .other));
    try testing.expectError(error.BadFlag, resolve(.{ .pp = 65 }, none, .other));
    try testing.expectEqual(@as(u32, 1), (try resolve(.{ .pp = 1 }, none, .other)).pipeline);
    try testing.expectError(error.BadFlag, resolve(.{ .tp = 2, .rank = 3 }, env(&.{ .{ "TF_GLM_WORLD", "4" }, .{ "TF_GLM_RANK", "3" } }), .glm));
}

test "a launch's mesh is its tensor group" {
    const m = try (try resolve(.{ .tp = 3, .rank = 2 }, env(&.{}), .other)).mesh();
    const g = m.group(.tensor);
    try testing.expectEqual(@as(u32, 2), g.rank);
    try testing.expectEqual(@as(u32, 3), g.world);
}
