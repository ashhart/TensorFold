//! Regression programs use the production planner, emitter, sharing and aligned workspace accounting.

const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const fp = @import("forward_prefill.zig");
const ws = @import("forward_prefill_k4.zig");
const check = @import("calls_check.zig");
const Config = @import("config.zig").Config;

const starts = [_]i64{ 0, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 290816 };

fn covered(a: std.mem.Allocator, cs: []const @import("block_calls.zig").Call, p: *const buffers.Plan, wp: *const buffers.Plan) !void {
    var need: buffers.Plan = .{ .a = a };
    try need.add(cs);
    for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
        const have = if (std.mem.indexOf(u8, role, fp.K4.tag) != null) wp.sizes.get(role) orelse 0 else p.sizes.get(role) orelse 0;
        if (have < bytes) std.debug.print("index scratch: {s}: need {d}, planned {d}\n", .{ role, bytes, have });
        try t.expect(have >= bytes);
    }
}

fn exercise(k4: bool, tbo: bool, budget: i64) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const o: block.Options = .{ .gm_v2 = .one, .limit = 299008, .rope_rows = 301056, .index_budget = budget, .taps = true, .pf_tbo = tbo, .branches = tbo };
    var f = @import("forward.zig").Forward.init(a, &cfg, &w, o, undefined, undefined);
    f.layers = &layers;
    const full = !k4 and !tbo;
    var state: fp.State = .{ .mode = if (full) .full else .replay, .k4 = k4 };
    f.prefill_state = &state;
    var p: buffers.Plan = .{ .a = a };
    var original: buffers.Plan = .{ .a = a };
    try fp.planPrograms(&f, &p, 2048, &layers);
    if (k4 or tbo) try ws.planWorkspace(&f, &original, 2048, &layers);
    if (tbo) try t.expectEqual(ws.planBytes(&original), try fp.workspaceBytes(&f, 2048));
    const row_sizes: [4]i64 = if (k4) .{ 2049, 3072, 4095, 4096 } else .{ 17, 1025, 2047, 2048 };
    for ([_]bool{ false, true }) |shared| {
        var wp: buffers.Plan = .{ .a = a };
        for (original.sizes.keys(), original.sizes.values()) |k, v| try wp.sizes.put(a, k, v);
        var keep: bp.Keep = .empty;
        if (shared) try fp.k4Trim(a, &wp, &p, &keep);
        var expected: u64 = 0;
        for (wp.sizes.keys(), wp.sizes.values()) |k, v| {
            if (std.mem.indexOf(u8, k, fp.K4.tag) != null) expected += std.mem.alignForward(u64, v, 256);
        }
        try t.expectEqual(expected, ws.planBytes(&wp));
        try t.expect(ws.planBytes(&original) >= expected);
        for (row_sizes) |n| for (starts) |start| {
            var tmp = std.heap.ArenaAllocator.init(t.allocator);
            defer tmp.deinit();
            const ta = tmp.allocator();
            const op = if (k4) fp.k4Options(o) else o;
            const plain = if (full) try bp.emitPrefill(ta, &cfg, &w, op, &layers, n, start, false) else try bp.emitReplay(ta, &cfg, &w, op, layers[0..21], n, start, .encoder);
            const cs = if (k4) try bp.ownCallsKeep(ta, plain, fp.K4.tag, if (shared) &keep else null) else plain;
            try covered(ta, cs, &p, &wp);
        };
        if (tbo) for ([_]i64{ 1024, 2046 }) |n| for ([_]i64{ 2048, 65536, 131072 }) |start| {
            var tmp = std.heap.ArenaAllocator.init(t.allocator);
            defer tmp.deinit();
            const ta = tmp.allocator();
            const op = if (k4) fp.k4Options(o) else o;
            const cs = try ws.wsPair(ta, try bp.emitTbo(ta, &cfg, &w, op, layers[0..21], n, n + 1, start), k4);
            // TBO sharing is disabled in production: its workspace owns both in-flight namespaces.
            if (!shared) try covered(ta, cs, &p, &wp);
        };
    }
}

test "index scratch: 88 ragged K4 cases with and without sharing, plus regular and TBO-B" {
    try exercise(true, false, 64 << 20);
    try exercise(false, false, 64 << 20);
    try exercise(false, true, 64 << 20);
    try exercise(true, true, 64 << 20);
}

test "index scratch: minimum materialised row-block budget boundary" {
    const boundary = 16 * 4096 * 12;
    for ([_]i64{ boundary - 1, boundary, boundary + 1 }) |budget| try exercise(true, false, budget);
}

test "index scratch: TBO-B materialisation missing from full-row endpoints is planned and priced" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const enc = layers[0..21];
    const o: block.Options = .{ .gm_v2 = .one, .limit = 299008, .rope_rows = 301056, .index_budget = 64 << 20, .taps = true, .pf_tbo = true, .branches = true };
    const op = fp.k4Options(o);
    var f = @import("forward.zig").Forward.init(a, &cfg, &w, o, undefined, undefined);
    f.layers = &layers;
    var state: fp.State = .{ .mode = .replay, .k4 = true };
    f.prefill_state = &state;
    var endpoints: buffers.Plan = .{ .a = a };
    const top = @max(0, try fp.lastStartWith(&f, op, enc, fp.K4.rows, .encoder) - fp.K4.rows);
    for ([_]i64{ 0, top }) |start| {
        try endpoints.add(try ws.wsPair(a, try bp.emitTbo(a, &cfg, &w, op, enc, fp.K4.rows, fp.K4.rows, start), true));
    }
    const scores_role = "L.k4~b~ix.scores";
    try t.expect(!endpoints.sizes.contains(scores_role));
    try t.expect(endpoints.sizes.contains("L.k4~b~ix.stream"));
    var ragged: buffers.Plan = .{ .a = a };
    try ragged.add(try ws.wsPair(a, try bp.emitTbo(a, &cfg, &w, op, enc, 1024, 1025, 2048), true));
    try t.expectEqual(@as(u64, 8_396_800), ragged.sizes.get(scores_role).?);
    var planned: buffers.Plan = .{ .a = a };
    try ws.planWorkspace(&f, &planned, 2048, &layers);
    const scores = planned.sizes.get(scores_role) orelse return error.TestUnexpectedResult;
    try t.expect(scores >= ragged.sizes.get(scores_role).?);
    // Encoder stops before the candidate source's indexer: neither candidate branch is reachable in B.
    try t.expect(!planned.sizes.contains("L.k4~b~ix.ckeys"));
    try t.expect(!planned.sizes.contains("L.k4~b~ix.bkeys"));
    const bytes = ws.planBytes(&planned);
    try t.expectEqual(bytes, try fp.workspaceBytes(&f, 2048));
    planned.sizes.getPtr(scores_role).?.* = 0;
    try t.expectEqual(std.mem.alignForward(u64, scores, 256), bytes - ws.planBytes(&planned));
    // A short context cannot stream; sampling the minimum pair must not allocate the streaming branch there.
    f.opts.limit = 4096;
    f.opts.rope_rows = 6144;
    state.k4 = false;
    var short: buffers.Plan = .{ .a = a };
    try ws.planWorkspace(&f, &short, 2048, &layers);
    try t.expect(short.sizes.contains(scores_role));
    try t.expect(!short.sizes.contains("L.k4~b~ix.stream"));
    try t.expectEqual(ws.planBytes(&short), try fp.workspaceBytes(&f, 2048));
}
