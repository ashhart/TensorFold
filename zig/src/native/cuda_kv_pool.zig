//! The native server's KV page pool: a paging family's plan and pages, and its cache split into slots and copied rows.
const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("core").kv_pool;
const budget = @import("cuda_memory.zig");
const Allocator = std.mem.Allocator;

pub const Plan = kv_pool.Plan;

/// A family whose other streams take pages (Nemotron's) names their layout; any other allocates a window each (null).
pub fn layout(loaded: anytype) ?kv_pool.Layout {
    return if (comptime @hasField(@TypeOf(loaded), "layout")) loaded.layout else null;
}

/// The pool beside the engine's own full-window sequence, which `model` counts once; null: `problem` says why not.
pub fn plan(comptime F: type, a: Allocator, l: kv_pool.Layout, room: u64, o: api.Open, window: u64, problem: *[]const u8) !?Plan {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    const want = budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
    return kv_pool.plan(l, .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows }) catch |e| {
        problem.* = switch (e) {
            error.NoStream => "the CUDA memory budget fits no stream beside the model",
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its other streams' fixed parts and a page each, and the CUDA memory budget leaves {d:.2} GiB: serve fewer, or free device memory", .{ o.lanes, toGib((o.lanes - 1) * (l.fixed_bytes + l.pageBytes())), toGib(room) }),
        };
        return null;
    };
}

/// The planned streams' sequences and pages made now; false: `problem` says why they could not be.
pub fn attach(comptime F: type, a: Allocator, ctx: *anyopaque, paged: ?Plan, problem: *[]const u8) !bool {
    if (comptime !@hasDecl(F, "attach")) return true;
    const p = paged orelse return true;
    F.attach(ctx, p.lanes - 1, p.pages) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA engine could not make its {d} streams' pages ({s})", .{ p.lanes - 1, @errorName(e) });
        return false;
    };
    return true;
}

/// The cache's snapshot slots and copied rows' bytes (`snaps` 0: a family without slots keeps its cache whole).
pub const Kept = struct { snaps: usize, copied: u64, snap_bytes: u64 };

pub fn kept(comptime F: type, loaded: anytype, paged: ?Plan, o: api.Open, room: u64, held: u64) Kept {
    if (comptime !@hasDecl(F, "snap_slots")) return .{ .snaps = 0, .copied = if (@hasDecl(F, "snap_save")) budget.cacheBytes(o.prompt_cache_gib, room, held) else 0, .snap_bytes = 0 };
    const b: u64 = loaded.snap_bytes;
    const s = if (paged) |p| kv_pool.split(p.snaps * b + p.copied, b, p.snaps) else kv_pool.split(budget.cacheBytes(o.prompt_cache_gib, room, held), b, named(o));
    return .{ .snaps = s.snaps, .copied = s.copied, .snap_bytes = b };
}

/// The store keeps a state a slot, the slots made now in one device buffer; a family without them, its own budget.
pub fn store(comptime F: type, s: *api.prompt_cache.Store, target: anytype, k: Kept) !void {
    if (comptime !@hasDecl(F, "snap_slots")) return;
    s.most = k.snaps;
    _ = try F.snap_slots(target, k.snaps);
}

/// The startup line's streams and cache.
pub fn text(comptime F: type, a: Allocator, stream_bytes: u64, l: ?kv_pool.Layout, paged: ?Plan, window: u64, k: Kept) ![2][]const u8 {
    const shape = if (paged) |p| try std.fmt.allocPrint(a, "the engine's own at a {d}-token window ({d:.2} GiB, in the model's), {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them, {d:.0} MiB fixed each", .{ window, toGib(stream_bytes), p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) }) else try std.fmt.allocPrint(a, "{d:.2} GiB each at a {d}-token window", .{ toGib(stream_bytes), window });
    const cache = if (k.copied == 0) ", no prompt cache" else if (comptime @hasDecl(F, "snap_slots")) try std.fmt.allocPrint(a, ", a prompt cache of {d} snapshot slots ({d:.1} GiB) and {d:.1} GiB of copied rows", .{ k.snaps, toGib(k.snaps * k.snap_bytes), toGib(k.copied) }) else try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache", .{toGib(k.copied)});
    return .{ shape, cache };
}

fn named(o: api.Open) ?usize {
    return if (o.keep) |k| @intCast(@max(k, 0)) else null;
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

test {
    std.testing.refAllDecls(@This());
}
