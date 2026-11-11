//! A CUDA family's place from the launch flags and its legacy rank variables (fabric/mesh/launch.zig), resolved once.
const std = @import("std");
const api = @import("engine_api");
const mesh = @import("mesh");
const settings = @import("sys").environment;
const glm = @import("glm_cuda").native;
const minimax = @import("minimax_m3_cuda").native;
const nemotron = @import("nemotron");

const Allocator = std.mem.Allocator;
const Family = mesh.launch.Family;

/// Which rank aliases `F` reads: only the CUDA families that had their own rank variables.
pub fn family(comptime F: type) Family {
    if (F == glm) return .glm;
    if (F == minimax) return .minimax;
    if (F == nemotron.native or F == nemotron.omni) return .nemotron;
    return .other;
}

/// `o` with `tp` and `rank` resolved and `launch_from` named, or null with `problem` set; before admission and `open`.
pub fn resolve(comptime F: type, a: Allocator, o: api.Open, problem: *[]const u8) !?api.Open {
    return resolveWith(a, o, family(F), settings.Process{}, problem);
}

pub fn resolveWith(a: Allocator, o: api.Open, fam: Family, lookup: anytype, problem: *[]const u8) !?api.Open {
    if (fam == .other) return o; // no aliases: the request's own tp and rank stand, checked by the family
    const flags: mesh.launch.Flags = .{
        .tp = if (o.given.tp) o.tp else null,
        .rank = if (o.given.rank) o.rank else null,
        .master = if (o.master.len > 0) o.master else null,
        .port = o.master_port,
    };
    const l = mesh.launch.resolve(flags, lookup, fam) catch |e| {
        // no world at all: GLM's own open refuses it (NoRank) where it always has, after the device and admission
        if (e == error.NoPlace) return unplaced(o);
        problem.* = try refusal(a, e, fam, o, lookup);
        return null;
    };
    for ([_]?[:0]const u8{ l.world_name, l.rank_name }) |name| if (name) |n| if (settings.retired(n, o.release)) |r| {
        problem.* = try std.fmt.allocPrint(a, "{s} was replaced by {s} in {s}; launch with {s}", .{ n, r.flag.?, r.retire.?, r.flag.? });
        return null;
    };
    var out = o;
    out.tp = l.tensor;
    out.rank = l.rank;
    out.launch_from = switch (l.from) {
        .flags => "--tp",
        .alias => try std.fmt.allocPrint(a, "{s}, an alias of --tp", .{l.world_name.?}),
        .mixed => try std.fmt.allocPrint(a, "--tp and {s}", .{l.world_name.?}),
        .default => "one rank",
    };
    if (l.world_name) |n| if (settings.rule(n).retire) |r|
        std.log.warn("{s} and {s} are aliases of --tp and --rank until {s}; launch with --tp {d} --rank {d}", .{ n, l.rank_name.?, r, l.tensor, l.rank });
    return out;
}

/// `o` as the flags left it, with no source: a family that needs a place refuses it in its own `open`.
fn unplaced(o: api.Open) api.Open {
    var out = o;
    out.launch_from = "";
    return out;
}

/// Each refusal names only the flags given and the alias values read, never a default the user did not give.
fn refusal(a: Allocator, e: mesh.launch.Error, fam: Family, o: api.Open, lookup: anytype) ![]const u8 {
    const n = mesh.launch.names(fam).?;
    const world = lookup.get(n.world) orelse "";
    const rank = lookup.get(n.rank) orelse "";
    return switch (e) {
        error.PartialAlias => std.fmt.allocPrint(a, "{s} and {s} go together: launch with --tp and --rank", .{ n.world, n.rank }),
        error.BadAlias => std.fmt.allocPrint(a, "{s}={s} and {s}={s}: whole numbers; launch with --tp and --rank", .{ n.world, world, n.rank, rank }),
        error.Disagree => if (o.given.tp and (std.fmt.parseInt(u32, world, 10) catch o.tp) != o.tp)
            std.fmt.allocPrint(a, "--tp {d} differs from {s}={s}: give one of them, or equal values", .{ o.tp, n.world, world })
        else
            std.fmt.allocPrint(a, "--rank {d} differs from {s}={s}: give one of them, or equal values", .{ o.rank, n.rank, rank }),
        error.BadFlag => badFlag(a, o, n, world, rank),
        error.NoPlace => unreachable, // resolveWith passes it on (unplaced)
        error.PipelineNotServed => "argument --pp: pipeline stages are not served yet (#2005); serve with --pp 1",
    };
}

/// The merged rank against the world's real source: --tp, the world alias, or --tp's default of 1.
fn badFlag(a: Allocator, o: api.Open, n: mesh.launch.Names, world: []const u8, rank: []const u8) ![]const u8 {
    const r = if (o.given.rank) o.rank else std.fmt.parseInt(u32, rank, 10) catch 0;
    if (o.given.tp) return std.fmt.allocPrint(a, "argument --rank: {d} is not below --tp {d}", .{ r, o.tp });
    if (world.len > 0) return std.fmt.allocPrint(a, "argument --rank: {d} is not below {s}={s}", .{ r, n.world, world });
    return std.fmt.allocPrint(a, "argument --rank: {d} is not below --tp 1 (its default; no --tp was given)", .{r});
}

const testing = std.testing;

const Map = struct {
    pairs: []const [2][]const u8,
    pub fn get(m: Map, name: []const u8) ?[]const u8 {
        for (m.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "GLM by its aliases, by flags or by both resolves alike; other families read no aliases" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const pair: Map = .{ .pairs = &.{ .{ "TF_GLM_WORLD", "2" }, .{ "TF_GLM_RANK", "1" } } };
    const none: Map = .{ .pairs = &.{} };
    const by_alias = (try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next" }, .glm, pair, &problem)).?;
    const by_flags = (try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .tp = 2, .rank = 1, .given = .{ .tp = true, .rank = true } }, .glm, none, &problem)).?;
    const by_both = (try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .tp = 2, .given = .{ .tp = true } }, .glm, pair, &problem)).?;
    for ([_]api.Open{ by_alias, by_flags, by_both }) |o| {
        try testing.expectEqual(@as(u32, 2), o.tp);
        try testing.expectEqual(@as(u32, 1), o.rank);
    }
    try testing.expectEqualStrings("TF_GLM_WORLD, an alias of --tp", by_alias.launch_from);
    try testing.expectEqualStrings("--tp", by_flags.launch_from);
    try testing.expectEqualStrings("--tp and TF_GLM_WORLD", by_both.launch_from);
    const mm: Map = .{ .pairs = &.{ .{ "TF_MINIMAX_WORLD", "2" }, .{ "TF_MINIMAX_RANK", "1" } } };
    const minimax_alias = (try resolveWith(a, .{ .dir = "d", .model_type = "minimax_m3_vl" }, .minimax, mm, &problem)).?;
    try testing.expectEqual(@as(u32, 1), minimax_alias.rank);
    try testing.expectEqualStrings("TF_MINIMAX_WORLD, an alias of --tp", minimax_alias.launch_from);
    try testing.expectEqual(Family.minimax, family(minimax));
    try testing.expectEqual(Family.glm, family(glm));
    try testing.expectEqual(Family.nemotron, family(nemotron.native));
    try testing.expectEqual(Family.nemotron, family(nemotron.omni));
    const other = (try resolveWith(a, .{ .dir = "d", .model_type = "m" }, .other, pair, &problem)).?;
    try testing.expectEqual(@as(u32, 1), other.tp);
    // a request built in code keeps its placement: no flag bits, no aliases, nothing to resolve
    const coded = (try resolveWith(a, .{ .dir = "d", .model_type = "m", .tp = 2, .rank = 1 }, .other, pair, &problem)).?;
    try testing.expectEqual(@as(u32, 2), coded.tp);
    try testing.expectEqual(@as(u32, 1), coded.rank);
}

test "refusals name the flags and the aliases; a retired alias names its flag" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const none: Map = .{ .pairs = &.{} };
    const unset = (try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next" }, .glm, none, &problem)).?;
    try testing.expectEqualStrings("", unset.launch_from); // GLM's open refuses it (NoRank), as it always has
    const pair: Map = .{ .pairs = &.{ .{ "TF_GLM_WORLD", "3" }, .{ "TF_GLM_RANK", "1" } } };
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .tp = 2, .given = .{ .tp = true } }, .glm, pair, &problem) == null);
    try testing.expectEqualStrings("--tp 2 differs from TF_GLM_WORLD=3: give one of them, or equal values", problem);
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .rank = 2, .given = .{ .rank = true } }, .glm, pair, &problem) == null);
    try testing.expectEqualStrings("--rank 2 differs from TF_GLM_RANK=1: give one of them, or equal values", problem);
    const two: Map = .{ .pairs = &.{ .{ "TF_GLM_WORLD", "2" }, .{ "TF_GLM_RANK", "2" } } };
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .rank = 2, .given = .{ .rank = true } }, .glm, two, &problem) == null);
    try testing.expectEqualStrings("argument --rank: 2 is not below TF_GLM_WORLD=2", problem);
    const partial: Map = .{ .pairs = &.{.{ "TF_GLM_WORLD", "3" }} };
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next" }, .glm, partial, &problem) == null);
    try testing.expectStringStartsWith(problem, "TF_GLM_WORLD and TF_GLM_RANK go together");
    // the retirement release is settings.alias_retire, set at landing; an earlier release still reads the alias
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .release = settings.alias_retire }, .glm, pair, &problem) == null);
    try testing.expectEqualStrings(try std.fmt.allocPrint(a, "TF_GLM_WORLD was replaced by --tp in {s}; launch with --tp", .{settings.alias_retire}), problem);
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .release = "0.0.1" }, .glm, pair, &problem) != null);
    // the values a launch gave, not defaults: a parsed world equal to --tp leaves the rank as the conflict
    const plus: Map = .{ .pairs = &.{ .{ "TF_GLM_WORLD", "+2" }, .{ "TF_GLM_RANK", "1" } } };
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .tp = 2, .rank = 0, .given = .{ .tp = true, .rank = true } }, .glm, plus, &problem) == null);
    try testing.expectEqualStrings("--rank 0 differs from TF_GLM_RANK=1: give one of them, or equal values", problem);
    try testing.expect(try resolveWith(a, .{ .dir = "d", .model_type = "glm5_next", .tp = 2, .given = .{ .tp = true } }, .glm, two, &problem) == null);
    try testing.expectEqualStrings("argument --rank: 2 is not below --tp 2", problem);
}
