//! Host tests of the multi-segment prefill run (block_prefill.emitMulti), run by `zig build test-dsv41-knobs`.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const ced = @import("ced.zig");
const buffers = @import("buffers.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 8192;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

fn options() block.Options {
    const pages = @divExact(limit, block.Pool.page);
    var o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048 };
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages, .split = false };
    return o;
}

fn isGlue(c: calls.Call, name: []const u8) bool {
    return c.glue and std.mem.eql(u8, c.name["glue.".len..], name);
}

/// The run's positions (rope's "w.mpos64") are a one-segment program's "w.pos64".
fn roleName(t: calls.Tensor) []const u8 {
    const r = switch (t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
    return if (std.mem.eql(u8, r, "w.mpos64")) "w.pos64" else r;
}

fn sameArg(x: calls.Arg, y: calls.Arg, offsets: bool) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            if (!std.mem.eql(u8, roleName(t), roleName(u)) or t.dt != u.dt) break :blk false;
            if (!offsets) break :blk true;
            break :blk t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        .none => true,
        .list => |l| l.len == y.list.len,
    };
}

/// The same launch: its name, grid and every argument (`offsets`: tensors' shapes, strides and offsets too).
fn same(x: calls.Call, y: calls.Call, offsets: bool) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.args.len != y.args.len) return false;
    if (offsets and !std.mem.eql(i64, &x.grid, &y.grid)) return false;
    for (x.args, y.args) |p, q| {
        // a run's fused core writes its rows of each group's plane: the plane's rows are the run's (checked apart)
        if (!offsets and std.mem.eql(u8, p.name, "OR")) continue;
        if (!sameArg(p.arg, q.arg, offsets)) return false;
    }
    return true;
}

/// The calls without the positions' glue (a run writes its own: mpositions, pf_seg).
fn body(a: std.mem.Allocator, cs: []const calls.Call) ![]const calls.Call {
    var out: std.ArrayList(calls.Call) = .empty;
    for (cs) |c| {
        if (isGlue(c, "positions") or isGlue(c, "mpositions") or isGlue(c, "pf_seg")) continue;
        try out.append(a, c);
    }
    return out.items;
}

fn named(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

const Layers = struct { all: [25]u32, enc: []const u32, dec: []const u32 };

fn layersOf(cfg: *const Config, l: *Layers) !void {
    for (&l.all, 0..) |*x, i| x.* = @intCast(i);
    const d = try ced.decoderStart(cfg);
    l.enc = l.all[0 .. d + 1];
    l.dec = l.all[d..];
}

test "a one-segment run is the one-segment program (full mode without the head, and CED's encoder pass)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var l: Layers = undefined;
    try layersOf(&cfg, &l);
    const o = options();
    for ([_]i64{ 33, 100, 127, 300, 2048 }) |n| for ([_]i64{ 0, 1000 }) |start| {
        const sp = [_]bp.Span{.{ .slot = 2, .start = start, .n = n }};
        const one = try body(a, try bp.emitPrefill(a, &cfg, &w, o, &l.all, n, start, false));
        const run = try body(a, try bp.emitMulti(a, &cfg, &w, o, &l.all, &sp, .whole));
        try testing.expectEqual(one.len, run.len);
        for (one, run) |x, y| try testing.expect(same(x, y, true));
        if (n <= 127) {
            // the decoder replay of a tail of n rows from R0 = start
            const dec1 = try body(a, try bp.emitReplay(a, &cfg, &w, o, l.dec, n, start, .decoder));
            const decr = try body(a, try bp.emitMulti(a, &cfg, &w, o, l.dec, &sp, .decoder));
            try testing.expectEqual(dec1.len, decr.len);
            for (dec1, decr) |x, y| try testing.expect(same(x, y, true));
        }
        const enc1 = try body(a, try bp.emitReplay(a, &cfg, &w, o, l.enc, n, start, .encoder));
        const encr = try body(a, try bp.emitMulti(a, &cfg, &w, o, l.enc, &sp, .encoder));
        try testing.expectEqual(enc1.len, encr.len);
        for (enc1, encr) |x, y| try testing.expect(same(x, y, true));
    };
}

test "a run of several segments: the rows' launches over all of them, each segment's CSA2 steps as its own program's, at its rows and on its slot" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var l: Layers = undefined;
    try layersOf(&cfg, &l);
    const o = options();
    const H: i64 = cfg.heads / o.world;
    const hd: i64 = cfg.head_dim;
    // slots at different positions and lengths: a staged segment past the ring, a ring-held one, one from 0
    const spans = [_]bp.Span{ .{ .slot = 3, .start = 4000, .n = 700 }, .{ .slot = 0, .start = 0, .n = 100 }, .{ .slot = 1, .start = 129, .n = 33 } };
    var total: i64 = 0;
    for (spans) |sp| total += sp.n;
    // the decoder replays: slots' tails from their R0 (a full 127-row tail, shorter prompts' tails)
    const tails = [_]bp.Span{ .{ .slot = 2, .start = 3873, .n = 127 }, .{ .slot = 0, .start = 0, .n = 64 }, .{ .slot = 3, .start = 0, .n = 33 } };
    for ([_]bp.Part{ .whole, .encoder, .decoder }) |part| {
        const layers = switch (part) {
            .encoder => l.enc,
            .decoder => l.dec,
            else => &l.all,
        };
        const segs: []const bp.Span = if (part == .decoder) &tails else &spans;
        total = 0;
        for (segs) |sp| total += sp.n;
        const run = try bp.emitMulti(a, &cfg, &w, o, layers, segs, part);
        // the run's per-row launches (outside its segments' CSA2 steps): an all-rows one-segment program's, in order
        const whole = try body(a, if (part != .whole) try bp.emitReplay(a, &cfg, &w, o, layers, total, segs[0].start, part) else try bp.emitPrefill(a, &cfg, &w, o, layers, total, segs[0].start, false));
        var at: usize = 0;
        var seg: ?usize = null;
        var cores: [3]std.ArrayList(calls.Call) = @splat(.empty);
        var entered: [3]usize = @splat(0);
        for (run) |c| {
            if (isGlue(c, "pf_seg")) {
                const j = c.args[0].arg.i;
                seg = if (j < 0) null else @intCast(j);
                if (seg) |s| entered[s] += 1;
                continue;
            }
            if (isGlue(c, "mpositions")) continue;
            if (seg) |s| {
                try cores[s].append(a, c);
                continue;
            }
            while (at < whole.len and !same(whole[at], c, false)) at += 1;
            if (at == whole.len) {
                std.debug.print("per-row launch {s} not in the all-rows program's order\n", .{c.name});
                return error.TestUnexpected;
            }
            at += 1;
        }
        // each segment's CSA2 steps: its own one-segment program's launches of those kinds, in order, at its rows
        var r0: i64 = 0;
        for (segs, 0..) |sp, j| {
            defer r0 += sp.n;
            try testing.expect(entered[j] > 0);
            const own = try body(a, if (part != .whole) try bp.emitReplay(a, &cfg, &w, o, layers, sp.n, sp.start, part) else try bp.emitPrefill(a, &cfg, &w, o, layers, sp.n, sp.start, false));
            var k: usize = 0;
            for (cores[j].items) |c| {
                while (k < own.len and !same(own[k], c, false)) k += 1;
                if (k == own.len) {
                    std.debug.print("segment {d}: {s} not in its own program's order\n", .{ j, c.name });
                    return error.TestUnexpected;
                }
                k += 1;
                // the fused core reads its rows of the run's q and writes its rows of each group's plane
                if (std.mem.eql(u8, c.name, "_fused")) {
                    try testing.expectEqual(r0 * H * hd * 2, named(c, "Q").?.t.offset);
                    try testing.expectEqual(total, named(c, "OR").?.i);
                    try testing.expectEqual(sp.n, c.grid[0]);
                }
                if (std.mem.eql(u8, c.name, "_kv_store") and std.mem.eql(u8, roleName(named(c, "LAT").?.t), "L.kv"))
                    try testing.expectEqual(r0 * hd * 2, named(c, "LAT").?.t.offset);
            }
        }
    }
}

test "a run refuses what one segment's program must run alone" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var l: Layers = undefined;
    try layersOf(&cfg, &l);
    const o = options();
    // a short segment (the decode kernels' mix), more rows than a prefill segment, the decoder pass
    try testing.expectError(error.Unsupported, bp.emitMulti(a, &cfg, &w, o, &l.all, &.{ .{ .slot = 0, .start = 0, .n = 100 }, .{ .slot = 1, .start = 0, .n = 16 } }, .whole));
    try testing.expectError(error.Unsupported, bp.emitMulti(a, &cfg, &w, o, &l.all, &.{ .{ .slot = 0, .start = 0, .n = 2000 }, .{ .slot = 1, .start = 0, .n = 100 } }, .whole));
    // a decoder tail of 32 rows or fewer (the split mHC programs: its replay alone is another kernel path)
    try testing.expectError(error.Unsupported, bp.emitMulti(a, &cfg, &w, o, l.dec, &.{ .{ .slot = 0, .start = 0, .n = 100 }, .{ .slot = 1, .start = 0, .n = 32 } }, .decoder));
}

test "every role a run reads is in the buffer plan, at its run's rows (forward_prefill.plan: the one-segment programs + planRuns)" {
    // regression: the runs' positions "w.mpos64" were in no planned program, so every run failed with Unbound
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var l: Layers = undefined;
    try layersOf(&cfg, &l);
    const o = options();
    const rows: i64 = 2048;
    var plan: buffers.Plan = .{ .a = a };
    try plan.add(try bp.emitPrefill(a, &cfg, &w, o, &l.all, rows, 0, true));
    try plan.add(try bp.emitReplay(a, &cfg, &w, o, l.enc, rows, 0, .encoder));
    try plan.add(try bp.emitReplay(a, &cfg, &w, o, l.dec, ced.keep, 0, .decoder));
    for ((try bp.planRuns(a, &cfg, &w, o, l.enc, l.dec, rows, .encoder)).items) |cs| try plan.add(cs);
    const runs = [_]struct { part: bp.Part, spans: []const bp.Span }{
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 1000 }, .{ .slot = 1, .start = 0, .n = 1048 } } },
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 17 }, .{ .slot = 1, .start = 0, .n = 300 }, .{ .slot = 2, .start = 0, .n = 33 } } },
        .{ .part = .decoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 127 }, .{ .slot = 1, .start = 0, .n = 127 }, .{ .slot = 2, .start = 0, .n = 127 }, .{ .slot = 3, .start = 0, .n = 64 } } },
        // the counts' widest layout (each 17-row segment's counts padded to the next multiple of 4)
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 17 }, .{ .slot = 1, .start = 0, .n = 17 }, .{ .slot = 2, .start = 0, .n = 17 }, .{ .slot = 3, .start = 0, .n = 17 }, .{ .slot = 4, .start = 0, .n = 17 }, .{ .slot = 5, .start = 0, .n = 17 }, .{ .slot = 6, .start = 0, .n = 17 }, .{ .slot = 7, .start = 0, .n = 17 }, .{ .slot = 8, .start = 0, .n = 17 }, .{ .slot = 9, .start = 0, .n = 17 }, .{ .slot = 10, .start = 0, .n = 17 }, .{ .slot = 11, .start = 0, .n = 17 }, .{ .slot = 12, .start = 0, .n = 17 }, .{ .slot = 13, .start = 0, .n = 17 }, .{ .slot = 14, .start = 0, .n = 17 }, .{ .slot = 15, .start = 0, .n = 1793 } } },
    };
    for (runs) |r| {
        const cs = try bp.emitMulti(a, &cfg, &w, o, if (r.part == .decoder) l.dec else l.enc, r.spans, r.part);
        var need: buffers.Plan = .{ .a = a };
        try need.add(cs);
        for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
            const have = plan.sizes.get(role) orelse {
                std.debug.print("role {s} of a run is in no planned program\n", .{role});
                return error.TestUnplanned;
            };
            if (have < bytes) {
                std.debug.print("role {s}: a run needs {d} bytes, the plan holds {d}\n", .{ role, bytes, have });
                return error.TestUnplanned;
            }
        }
    }
}

/// The first Triton tensor argument in `cs` not 16-byte aligned from its role's base, or null (Triton specializes).
pub fn misaligned(cs: []const calls.Call) ?struct { call: []const u8, arg: []const u8, offset: i64 } {
    for (cs) |c| {
        if (!c.triton) continue;
        for (c.args) |x| switch (x.arg) {
            .t, .opaque_table => |t| if (@mod(t.offset, 16) != 0) return .{ .call = c.name, .arg = x.name, .offset = t.offset },
            else => {},
        };
    }
    return null;
}

test "every tensor a piece run or replay run hands Triton is 16-byte aligned (Triton's specialization, aot-needs' key)" {
    // regression: a run's second segment read selection counts at row0 * 4, leaving _fused's CNT not 16-aligned
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var l: Layers = undefined;
    try layersOf(&cfg, &l);
    const o = options();
    // first rows at every residue mod 4, staged segments (past 129 rows), the counts' widest layout
    var widest: [16]bp.Span = undefined;
    for (widest[0..15], 0..) |*x, i| x.* = .{ .slot = @intCast(i), .start = 0, .n = 17 };
    widest[15] = .{ .slot = 15, .start = 0, .n = 2048 - 15 * 17 };
    const runs = [_]struct { part: bp.Part, spans: []const bp.Span }{
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 17 }, .{ .slot = 1, .start = 0, .n = 18 }, .{ .slot = 2, .start = 700, .n = 19 }, .{ .slot = 3, .start = 0, .n = 20 } } },
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 0, .n = 159 }, .{ .slot = 1, .start = 4000, .n = 1497 } } },
        .{ .part = .encoder, .spans = &.{ .{ .slot = 0, .start = 2048, .n = 1501 }, .{ .slot = 1, .start = 0, .n = 131 }, .{ .slot = 2, .start = 0, .n = 33 } } },
        .{ .part = .whole, .spans = &.{ .{ .slot = 0, .start = 0, .n = 299 }, .{ .slot = 1, .start = 0, .n = 499 }, .{ .slot = 2, .start = 0, .n = 699 } } },
        .{ .part = .encoder, .spans = &widest },
        .{ .part = .decoder, .spans = &.{ .{ .slot = 0, .start = 33, .n = 127 }, .{ .slot = 1, .start = 0, .n = 33 }, .{ .slot = 2, .start = 0, .n = 66 }, .{ .slot = 3, .start = 900, .n = 127 } } },
    };
    for (runs) |r| {
        const layers = switch (r.part) {
            .encoder => l.enc,
            .decoder => l.dec,
            else => &l.all,
        };
        const cs = try bp.emitMulti(a, &cfg, &w, o, layers, r.spans, r.part);
        if (misaligned(cs)) |m| {
            std.debug.print("{s} {s}: offset {d} is not 16-byte aligned\n", .{ m.call, m.arg, m.offset });
            return error.TestMisaligned;
        }
    }
}
