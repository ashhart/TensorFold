//! Decode windows past 16 rows (TF_DSV41_ROWS_CAP) on the real emitter, host only: mHC launches and router config.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const graphs = @import("graphs.zig");
const rowmode = @import("rowmode.zig");
const dspark_emit = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;

// TF_DSV41_MHC_PFDEC on the same emitters
test {
    _ = @import("mhc_pfdec_test.zig");
    _ = @import("coef_late_test.zig");
    _ = @import("ibound_test.zig");
    _ = @import("dev_arena_test.zig");
    _ = @import("x3ld_epi_test.zig");
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn servedOptions(limit: i64) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn arg(c: calls.Call, name: []const u8) calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    std.debug.print("{s}: no argument {s}\n", .{ c.name, name });
    @panic("argument");
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t => |t| t.role.buf,
        else => "",
    };
}

/// One expected mHC site (the call and its launch parameters).
const Site = struct { post: bool, collapse: i64, mix: bool, split: bool, tap: bool, in_place: bool };

fn expected(a: std.mem.Allocator, cfg: *const Config, n: i64) ![]Site {
    var out: std.ArrayList(Site) = .empty;
    const alt = n <= 32; // mhc.out_of_place
    const split_ok = n <= 32; // mhc.split_rows
    for (0..cfg.layers) |i| {
        const L: u32 = @intCast(i);
        const engram = std.mem.indexOfScalar(u32, cfg.engram_layers.items(), L) != null;
        const aux = std.mem.indexOfScalar(u32, cfg.dspark_targets.items(), L) != null;
        if (i == 0) {
            try out.append(a, .{ .post = false, .collapse = 1, .mix = true, .split = split_ok, .tap = false, .in_place = false });
        } else if (engram) {
            try out.append(a, .{ .post = true, .collapse = 0, .mix = false, .split = false, .tap = false, .in_place = true });
            try out.append(a, .{ .post = false, .collapse = 2, .mix = true, .split = split_ok, .tap = false, .in_place = false });
        } else {
            try out.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = split_ok and alt, .tap = aux, .in_place = !alt });
        }
        try out.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = split_ok and alt, .tap = false, .in_place = !alt });
    }
    try out.append(a, .{ .post = true, .collapse = 2, .mix = false, .split = false, .tap = false, .in_place = true });
    return out.items;
}

fn expectSites(a: std.mem.Allocator, cfg: *const Config, cs: []const calls.Call, n: i64) !void {
    const want = try expected(a, cfg, n);
    var k: usize = 0;
    var finishes: usize = 0;
    var mixes: usize = 0;
    for (cs, 0..) |c, ci| {
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_mhc_cuda")) {
            std.debug.print("n {d}: mhc_cuda call at {d}\n", .{ n, ci });
            return error.TestUnexpectedResult;
        }
        if (std.mem.eql(u8, c.name, "_finish_k")) {
            finishes += 1;
            // after a mixing site: COEF; after the final: not
            try testing.expect(k > 0);
            try testing.expectEqual(want[k - 1].mix, arg(c, "COEF").b);
            try testing.expectEqual(@as(i64, n), c.grid[0]);
            continue;
        }
        if (!std.mem.eql(u8, c.name, "_site")) continue;
        if (k >= want.len) return error.TestUnexpectedResult;
        const s = want[k];
        k += 1;
        if (s.mix) mixes += 1;
        errdefer std.debug.print("n {d}: site {d} (call {d}) want {any}\n", .{ n, k - 1, ci, s });
        try testing.expectEqual(s.post, arg(c, "POST_ON").b);
        try testing.expectEqual(s.collapse, arg(c, "COLLAPSE").i);
        try testing.expectEqual(s.mix, arg(c, "MIX").b);
        try testing.expectEqual(s.split, arg(c, "SPLIT").b);
        try testing.expectEqual(s.tap, arg(c, "TAP_ON").b);
        try testing.expectEqual(@as(i64, 16), arg(c, "BM").i);
        try testing.expectEqual(n, arg(c, "R").i);
        try testing.expectEqual([3]i64{ std.math.divCeil(i64, n, 16) catch unreachable, 40, if (s.split) 4 else 1 }, c.grid);
        if (s.post) try testing.expectEqual(s.in_place, std.mem.eql(u8, roleOf(arg(c, "X")), roleOf(arg(c, "XOUT"))));
    }
    try testing.expectEqual(want.len, k);
    try testing.expectEqual(mixes + 1, finishes);
}

fn expectRouters(cs: []const calls.Call) !usize {
    var m: usize = 0;
    for (cs) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) {
        // (row tile, warps, experts a warp) after (x, gate, bias, pick, wts, logits, cnt, topk, slots, scale)
        try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
        try testing.expectEqual(@as(i64, 8), c.args[11].arg.i);
        try testing.expectEqual(@as(i64, 2), c.args[12].arg.i);
        // the router groups only on the narrow kernel (<= 16 rows): `group` is off, so upstream's group runs after
        try testing.expect(!c.args[19].arg.b);
        try testing.expectEqual(@as(i64, 0), c.args[18].arg.i);
        m += 1;
    };
    var g: usize = 0;
    for (cs) |c| g += @intFromBool(std.mem.eql(u8, c.name, "tensorfold_exl3_experts_v1.group"));
    try testing.expectEqual(m, g);
    return m;
}

test "router group/rotation fusion removes one wide call per MoE and keeps narrow R1 unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const off = servedOptions(1 << 20);
    var on = off;
    on.router_group_rot = true;
    for ([_]i64{ 16, 17, 24, 32, 48, 64 }) |n| {
        const plain = try block.emit(a, &cfg, &w, off, &backbone, n, 4080, true);
        const fused = try block.emit(a, &cfg, &w, on, &backbone, n, 4080, true);
        const removed: usize = if (n <= 16) 0 else cfg.layers;
        try testing.expectEqual(plain.len - removed, fused.len);
        var pi: usize = 0;
        var combined: usize = 0;
        for (fused) |c| {
            if (std.mem.eql(u8, c.name, "tensorfold_exl3_experts_v1.group_rot_in")) {
                const group = plain[pi];
                const rot = plain[pi + 1];
                try testing.expectEqualStrings("tensorfold_exl3_experts_v1.group", group.name);
                try testing.expectEqualStrings("tensorfold_exl3_experts_v1.rot_in", rot.name);
                try testing.expectEqual(@as(usize, 14), c.args.len);
                for (rot.args, c.args[0..11]) |want, got| try testing.expectEqualDeep(want, got);
                for (group.args[1..4], c.args[11..14]) |want, got| try testing.expectEqualDeep(want, got);
                combined += 1;
                pi += 2;
            } else {
                try testing.expectEqualDeep(plain[pi], c);
                pi += 1;
            }
        }
        try testing.expectEqual(removed, combined);
        try testing.expectEqual(plain.len, pi);
        const transformed = try rowmode.transform(a, fused, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = on.pool.?.pts });
        var retained: usize = 0;
        for (transformed) |c| retained += @intFromBool(std.mem.eql(u8, c.name, "tensorfold_exl3_experts_v1.group_rot_in"));
        try testing.expectEqual(combined, retained);
    }
}

test "wide rows: decode windows of 17-64 rows take Python's > 16-row mHC and router path, and row mode takes them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    const o = servedOptions(limit);
    const s: graphs.Settings = .{};
    for ([_]i64{ 17, 20, 24, 32, 33, 48, 64 }) |n| {
        for ([_]u64{ 5000, 300_000, @intCast(limit - 1) }) |last| {
            const ctx = graphs.bucketOf(last, s.bucket, s.grow);
            const start = graphs.emitStart(@intCast(n), ctx, s, @intCast(limit));
            const one = try block.emit(a, &cfg, &w, o, &backbone, n, start, true);
            try expectSites(a, &cfg, one, n);
            try testing.expectEqual(@as(usize, cfg.layers), try expectRouters(one));
            const rows = try rowmode.transform(a, one, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = o.pool.?.pts });
            try testing.expect(rows.len > 0);
        }
    }
}

test "wide rows: 16-row windows keep mhc_cuda and the narrow router" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const o = servedOptions(1 << 20);
    const one = try block.emit(a, &cfg, &w, o, &backbone, 16, 4080, true);
    var cuda: usize = 0;
    for (one) |c| {
        if (std.mem.eql(u8, c.name, "tf_dsv41_mhc_cuda_v1.run")) cuda += 1;
        try testing.expect(!std.mem.eql(u8, c.name, "_finish_k"));
        if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) {
            try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
            try testing.expectEqual(@as(i64, 2), c.args[11].arg.i);
            try testing.expectEqual(@as(i64, 0), c.args[12].arg.i);
            try testing.expect(c.args[19].arg.b); // grouped in the narrow kernel's tail (R1's fold)
        }
        try testing.expect(!std.mem.eql(u8, c.name, "tensorfold_exl3_experts_v1.group"));
    }
    try testing.expect(cuda > 0);
}

test "wide rows: a 4-slot DSpark pass (20 rows) as Python's drafter at pad_slots(4)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    const o = servedOptions(1 << 20);
    const cs = try dspark_emit.emitPassSlots(a, &cfg, &w, o, cfg.dspark_block, 4, 4);
    // the drafter's device pass: block 0's site (collapse 1, split, no post), in-place boundaries, the final (no mix)
    var sites: std.ArrayList(Site) = .empty;
    for (0..cfg.mtp_layers) |b| {
        if (b == 0) try sites.append(a, .{ .post = false, .collapse = 1, .mix = true, .split = true, .tap = false, .in_place = false }) else try sites.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = false, .tap = false, .in_place = true });
        try sites.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = false, .tap = false, .in_place = true });
    }
    try sites.append(a, .{ .post = true, .collapse = 2, .mix = false, .split = false, .tap = false, .in_place = true });
    var k: usize = 0;
    for (cs) |c| {
        try testing.expect(!std.mem.startsWith(u8, c.name, "tf_dsv41_mhc_cuda"));
        if (!std.mem.eql(u8, c.name, "_site")) continue;
        const s = sites.items[k];
        k += 1;
        try testing.expectEqual(s.post, arg(c, "POST_ON").b);
        try testing.expectEqual(s.collapse, arg(c, "COLLAPSE").i);
        try testing.expectEqual(s.mix, arg(c, "MIX").b);
        try testing.expectEqual(s.split, arg(c, "SPLIT").b);
        try testing.expectEqual(@as(i64, 20), arg(c, "R").i);
    }
    try testing.expectEqual(sites.items.len, k);
    try testing.expectEqual(cfg.mtp_layers, try expectRouters(cs));
}

test "KV norm/store removes only R1 SWA producers and row mode binds slot writes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const off = servedOptions(1 << 20);
    var on = off;
    on.kv_norm_store = true;
    for ([_]i64{ 1, 2, 16, 17, 24, 64 }) |n| {
        const plain = try block.emit(a, &cfg, &w, off, &backbone, n, 4080, true);
        const fused = try block.emit(a, &cfg, &w, on, &backbone, n, 4080, true);
        try @import("kv_glue_emit_test.zig").compare(plain, fused, cfg.layers);
        const transformed = try rowmode.transform(a, fused, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = on.pool.?.pts });
        var count: usize = 0;
        for (transformed) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_kv_glue_v1.norm_store")) {
            try testing.expect(c.args[14].arg.b);
            try testing.expectEqual(@as(i64, n), c.args[5].arg.t.shape[0]);
            try testing.expectEqual(@as(i64, @intCast(@import("rowtab.zig").colOffset(.pos, 64))), c.args[5].arg.t.offset);
            try testing.expectEqual(@as(i64, @intCast(@import("rowtab.zig").colOffset(.wslot, 64))), c.args[6].arg.t.offset);
            try testing.expectEqual(@as(i64, 4 * 2 * cfg.window), c.args[3].arg.t.shape[0]);
            count += 1;
        };
        try testing.expectEqual(@as(usize, cfg.layers), count);
    }
    var legacy = off;
    legacy.r1 = false;
    var legacy_on = legacy;
    legacy_on.kv_norm_store = true;
    try testing.expectEqualDeep(try block.emit(a, &cfg, &w, legacy, &backbone, 1, 4080, true), try block.emit(a, &cfg, &w, legacy_on, &backbone, 1, 4080, true));
}
