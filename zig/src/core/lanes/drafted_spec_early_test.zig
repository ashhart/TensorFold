//! Host tests for Drafted speculate_early overlap with candidate verify trees, non-prefix absorbs, and cut fallback.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const Stream = sm.Stream;
const Held = sm.Held;
const Sampling = @import("sampling.zig").Sampling;
const fake = @import("fake.zig");
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;

const gpa = std.testing.allocator;

const TreeSpecDrafter = struct {
    const LaneState = struct {
        ctx: std.ArrayList(u32) = .empty,
        tokens: [15]u32 = @splat(0),
        parents: [15]i32 = @splat(-1),
        chances: [15]f64 = @splat(0),
        grown: usize = 0,
    };
    const SpecRec = struct {
        stream: *Stream,
        start: u64,
        rows_len: usize,
        rows: [16]u32 = @splat(0),
        has_rows: bool = false,
        pending: u32 = 0,
        anchor: u64 = 0,
        has_hold: bool = false,
    };

    lanes: std.AutoHashMapUnmanaged(*Stream, LaneState) = .empty,
    spec_recs: [8]SpecRec = undefined,
    spec_n: usize = 0,
    spec_active: bool = false,
    spec_launches: u32 = 0,
    spec_hits: u32 = 0,
    spec_misses: u32 = 0,
    non_prefix_absorbs: u32 = 0,
    rounds: u32 = 0,

    fn deinit(d: *TreeSpecDrafter) void {
        var it = d.lanes.valueIterator();
        while (it.next()) |l| l.ctx.deinit(gpa);
        d.lanes.deinit(gpa);
    }

    fn drafter(d: *TreeSpecDrafter) dr.Drafter {
        return .{
            .kind = "TreeSpec",
            .ptr = d,
            .vtable = &.{
                .taps = tapsFn,
                .facts = factsFn,
                .open = openFn,
                .absorb = absorbFn,
                .hold = holdFn,
                .draft_round = draftRoundFn,
                .speculate_early = speculateEarlyFn,
                .held = heldFn,
                .release = releaseFn,
                .tree = treeFn,
                .chances = chancesFn,
            },
        };
    }

    fn cast(ptr: *anyopaque) *TreeSpecDrafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn tapsFn(_: *anyopaque) []const u32 {
        return &.{ 5, 23 };
    }
    fn factsFn(_: *anyopaque) dr.Facts {
        return .{ .fixed_block = true, .depth = 7, .step_ms = 0.05, .plain_guard = false, .batched = true, .tree_nodes = 15 };
    }
    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        try cast(ptr).lanes.put(gpa, s, .{});
    }
    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        d.spec_active = false;
        if (d.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            l.ctx.deinit(gpa);
        }
    }

    fn applyAbsorbs(d: *TreeSpecDrafter, items: []const dr.Absorb) !void {
        for (items) |it| {
            const lane = d.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            const n = if (it.rows) |rs| rs.len else it.features.rows;
            if (it.rows != null) d.non_prefix_absorbs += 1;
            if (it.follow.len != n) return error.BadFollow;
            while (lane.ctx.items.len < it.start) try lane.ctx.append(gpa, 0);
            lane.ctx.shrinkRetainingCapacity(@intCast(it.start));
            try lane.ctx.appendSlice(gpa, it.follow);
        }
    }

    fn buildTree(d: *TreeSpecDrafter, it: dr.Hold) !void {
        const lane = d.lanes.getPtr(it.stream) orelse return error.UnknownStream;
        // Branch 2 carries fake.next at root so non-prefix paths [0, 2, ...] are accepted and compacted.
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(gpa);
        try ctx.appendSlice(gpa, it.tokens);
        while (ctx.items.len < it.position) try ctx.append(gpa, it.pending);
        ctx.shrinkRetainingCapacity(@intCast(it.position));
        ctx.items[it.position - 1] = it.pending;
        const t0 = fake.next(ctx.items, it.sampling, it.position);
        const wrong0 = (t0 + 1) % 97;
        try ctx.append(gpa, t0);
        const t1 = fake.next(ctx.items, it.sampling, it.position + 1);
        const wrong1 = (t1 + 2) % 97;
        try ctx.append(gpa, t1);
        const t2 = fake.next(ctx.items, it.sampling, it.position + 2);
        // Nodes: 0=wrong0 (-1), 1=t0 (-1), 2=wrong1 (1), 3=t1 (1), 4=t2 (3), then chain up to 15.
        const cap = @min(@as(usize, 15), @max(@as(usize, 1), @as(usize, @intCast(@max(1, it.stream.draftRoom() - 1)))));
        var k: usize = 0;
        const seed_toks = [_]u32{ wrong0, t0, wrong1, t1, t2 };
        const seed_pars = [_]i32{ -1, -1, 1, 1, 3 };
        const seed_ch = [_]f64{ 0.90, 0.85, 0.70, 0.65, 0.55 };
        while (k < cap and k < seed_toks.len) : (k += 1) {
            lane.tokens[k] = seed_toks[k];
            lane.parents[k] = seed_pars[k];
            lane.chances[k] = seed_ch[k];
        }
        while (k < cap) : (k += 1) {
            lane.tokens[k] = @intCast((t2 + k * 7) % 97);
            lane.parents[k] = @intCast(k - 1);
            lane.chances[k] = @max(0.05, 0.50 - 0.03 * @as(f64, @floatFromInt(k)));
        }
        lane.grown = k;
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        d.spec_active = false;
        try d.applyAbsorbs(items);
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        d.spec_active = false;
        for (items) |it| try d.buildTree(it);
    }

    fn specMatches(d: *const TreeSpecDrafter, absorbs: []const dr.Absorb, holds: []const dr.Hold) bool {
        if (!d.spec_active or absorbs.len != d.spec_n) return false;
        var h_idx: usize = 0;
        for (absorbs, d.spec_recs[0..d.spec_n]) |it, want| {
            if (it.stream != want.stream or it.start != want.start) return false;
            const n = if (it.rows) |rs| rs.len else it.features.rows;
            if (n != want.rows_len) return false;
            if (it.rows) |rs| {
                if (want.has_rows) {
                    if (!std.mem.eql(u32, rs, want.rows[0..want.rows_len])) return false;
                } else return false;
            } else if (want.has_rows) return false;
            if (want.has_hold) {
                if (h_idx >= holds.len) return false;
                const h = holds[h_idx];
                if (h.stream != want.stream or h.pending != want.pending or h.position - 1 != want.anchor) return false;
                h_idx += 1;
            }
        }
        return h_idx == holds.len;
    }

    fn speculateEarlyFn(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (absorbs.len > d.spec_recs.len or holds.len == 0) return;
        d.spec_n = absorbs.len;
        var h_idx: usize = 0;
        for (absorbs, 0..) |it, i| {
            const n = if (it.rows) |rs| rs.len else it.features.rows;
            var rec: SpecRec = .{ .stream = it.stream, .start = it.start, .rows_len = n, .has_rows = it.rows != null };
            if (it.rows) |rs| @memcpy(rec.rows[0..rs.len], rs);
            if (h_idx < holds.len and holds[h_idx].stream == it.stream) {
                rec.has_hold = true;
                rec.pending = holds[h_idx].pending;
                rec.anchor = holds[h_idx].position - 1;
                h_idx += 1;
            }
            d.spec_recs[i] = rec;
        }
        d.spec_active = true;
        d.spec_launches += 1;
    }

    fn draftRoundFn(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        d.rounds += 1;
        if (d.spec_active) {
            if (d.specMatches(absorbs, holds)) d.spec_hits += 1 else d.spec_misses += 1;
            d.spec_active = false;
        }
        try d.applyAbsorbs(absorbs);
        for (holds) |it| try d.buildTree(it);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, dst| {
            const lane = d.lanes.getPtr(s) orelse return error.UnknownStream;
            @memcpy(dst, lane.tokens[0..dst.len]);
        }
    }

    fn treeFn(ptr: *anyopaque, s: *Stream, alloc: std.mem.Allocator) anyerror!?Held {
        const d = cast(ptr);
        const lane = d.lanes.getPtr(s) orelse return null;
        if (lane.grown == 0) return null;
        const toks = try alloc.dupe(u32, lane.tokens[0..lane.grown]);
        errdefer alloc.free(toks);
        return .{ .count = @intCast(lane.grown), .tokens = toks, .parents = try alloc.dupe(i32, lane.parents[0..lane.grown]) };
    }

    fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        const lane = d.lanes.getPtr(s) orelse return false;
        if (lane.grown == 0 or out.len > lane.grown) return false;
        @memcpy(out, lane.chances[0..out.len]);
        return true;
    }
};

fn makeCosts(out: *[16]Cost) void {
    for (out, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 10.0 + 0.05 * @as(f64, @floatFromInt(w)) };
}

const TapTarget = struct {
    fake: fake.Fake = .{ .gpa = gpa },
    vtable: @import("backend.zig").Backend.VTable = undefined,

    fn backend(t: *TapTarget) @import("backend.zig").Backend {
        var b = t.fake.backend();
        t.vtable = b.vtable.*;
        t.vtable.features = featuresFn;
        b.vtable = &t.vtable;
        return b;
    }

    fn featuresFn(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!@import("backend.zig").Features {
        const f: *fake.Fake = @ptrCast(@alignCast(ptr));
        if (f.lanes.getPtr(s)) |l| {
            if (start == l.base and count <= l.rows.items.len) {
                return .{ .buffer = @intFromPtr(l.rows.items.ptr), .offset = 0, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
            }
        }
        return f.backend().features(s, taps, start, count);
    }
};

fn runCase(specs: []const sm.Spec, use_drafter: bool) !struct { tokens: [][]u32, hits: u32, misses: u32, non_prefix: u32 } {
    var costs: [16]Cost = undefined;
    makeCosts(&costs);
    var target: TapTarget = .{};
    defer target.fake.deinit();
    var head: TreeSpecDrafter = .{};
    defer head.deinit();
    var wrapped: Drafted = undefined;
    if (use_drafter) wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer if (use_drafter) wrapped.deinit();
    const base_model: Model = .{ .exact_width = 16, .hidden_rows = true, .batch_rows = 32, .max_streams = 4, .window_costs = &costs };
    const model = if (use_drafter) wrapped.facts(base_model) else base_model;
    var cfg = try Config.init(gpa, model, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var eng = Engine.init(gpa, &cfg, if (use_drafter) wrapped.backend() else target.backend(), clock.clock());
    defer eng.deinit();

    const streams = try gpa.alloc(Stream, specs.len);
    defer gpa.free(streams);
    var inited: usize = 0;
    defer for (streams[0..inited]) |*s| s.deinit(gpa);
    for (specs, streams) |sp, *s| {
        s.* = try Stream.init(gpa, sp);
        inited += 1;
        try eng.addStream(s);
    }
    while (eng.activeCount() > 0) try eng.step();
    const out = try gpa.alloc([]u32, specs.len);
    for (streams, out) |*s, *dst| dst.* = try gpa.dupe(u32, s.emitted());
    return .{ .tokens = out, .hits = head.spec_hits, .misses = head.spec_misses, .non_prefix = head.non_prefix_absorbs };
}

test "speculate_early with 15-node candidate trees matches plain decode bit-for-bit across T=0, T=1, c=1, c=2" {
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };
    const sampled: Sampling = .{ .seed = 0x1234_5678, .temperature = 1.0, .top_k = 16, .top_p = 0.95 };
    const cases = [_][]const sm.Spec{
        &.{.{ .id = "lone_t0", .prompt = &p1, .max_new = 24 }},
        &.{.{ .id = "lone_t1", .prompt = &p1, .max_new = 24, .sampling = sampled }},
        &.{ .{ .id = "c2_a", .prompt = &p1, .max_new = 20 }, .{ .id = "c2_b", .prompt = &p2, .max_new = 22, .sampling = sampled } },
    };
    for (cases) |specs| {
        const plain = try runCase(specs, false);
        defer {
            for (plain.tokens) |t| gpa.free(t);
            gpa.free(plain.tokens);
        }
        const tree = try runCase(specs, true);
        defer {
            for (tree.tokens) |t| gpa.free(t);
            gpa.free(tree.tokens);
        }
        for (plain.tokens, tree.tokens) |want, got| try std.testing.expectEqualSlices(u32, want, got);
        try std.testing.expect(tree.hits > 0);
        try std.testing.expect(tree.non_prefix > 0);
    }
}

test "speculate_early handles think-budget cut miss cleanly without changing output tokens" {
    const p = [_]u32{ 4, 5, 6, 7, 8, 9 };
    const close = [_]u32{ 88, 89 };
    const specs = [_]sm.Spec{.{ .id = "cut", .prompt = &p, .max_new = 20, .think_budget = 5, .think_close = &close, .think_end = 89 }};
    const plain = try runCase(&specs, false);
    defer {
        for (plain.tokens) |t| gpa.free(t);
        gpa.free(plain.tokens);
    }
    const tree = try runCase(&specs, true);
    defer {
        for (tree.tokens) |t| gpa.free(t);
        gpa.free(tree.tokens);
    }
    try std.testing.expectEqualSlices(u32, plain.tokens[0], tree.tokens[0]);
    try std.testing.expect(tree.misses >= 1);
}

test "speculate_early skips plain rounds where allocator grants zero drafts without launching wasted holds" {
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };
    const specs = [_]sm.Spec{
        .{ .id = "plain_a", .prompt = &p1, .max_new = 12 },
        .{ .id = "plain_b", .prompt = &p2, .max_new = 12 },
    };
    const plain = try runCase(&specs, false);
    defer {
        for (plain.tokens) |t| gpa.free(t);
        gpa.free(plain.tokens);
    }
    // Steep shared costs: width <= 2 costs 10 ms, extra draft rows cost 1000 ms -> plain rounds (rows == 1).
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = if (w <= 2) 10.0 else 1000.0 * @as(f64, @floatFromInt(w)) };
    var target: TapTarget = .{};
    defer target.fake.deinit();
    var head: TreeSpecDrafter = .{};
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const base_model: Model = .{ .exact_width = 16, .hidden_rows = true, .batch_rows = 32, .max_streams = 4, .window_costs = &costs };
    var cfg = try Config.init(gpa, wrapped.facts(base_model), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var eng = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer eng.deinit();
    var streams: [2]Stream = undefined;
    for (specs, &streams) |sp, *s| {
        s.* = try Stream.init(gpa, sp);
        try eng.addStream(s);
    }
    defer for (&streams) |*s| s.deinit(gpa);
    while (eng.activeCount() > 0) try eng.step();
    for (plain.tokens, &streams) |want, *s| try std.testing.expectEqualSlices(u32, want, s.emitted());
    try std.testing.expectEqual(@as(u32, 0), head.spec_launches);
    try std.testing.expectEqual(@as(u32, 0), head.spec_misses);
}
