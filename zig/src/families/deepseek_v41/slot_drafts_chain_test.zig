//! Slot drafts' host tests of the overlapped ingest, the device chain and the batched main_proj.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const config = @import("config.zig");
const Config = config.Config;
const graphs = @import("graphs.zig");
const rowmode = @import("rowmode.zig");
const run = @import("run.zig");
const dsd = @import("dspark_dev.zig");
const dsk = @import("draft/dspark.zig");
const fixture = @import("slot_drafts_fixture.zig");
const Fixture = fixture.Fixture;
const widths = fixture.widths;
const equalArgs = fixture.equalArgs;
const identical = fixture.identical;

// -- TF_DSV41_DRAFT_OVERLAP / TF_DSV41_DRAFT_CHAIN ----------------------------------------------------------------

fn sameArgSide(x: calls.Arg, y: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            const u = switch (y) {
                .t, .opaque_table => |v| v,
                else => return error.TestUnexpectedResult,
            };
            try testing.expectEqual(t.dt, u.dt);
            try testing.expectEqualSlices(i64, t.shape, u.shape);
            try testing.expectEqualSlices(i64, t.stride, u.stride);
            try testing.expectEqual(t.offset, u.offset);
            switch (t.role) {
                .buf => |r| if (emit.sideRole(r)) {
                    try testing.expect(std.mem.startsWith(u8, u.role.buf, "w.ovl."));
                    try testing.expectEqualStrings(r, u.role.buf["w.ovl.".len..]);
                } else try testing.expectEqualStrings(r, u.role.buf),
                .weight => |w| try testing.expectEqualStrings(w, u.role.weight),
                .empty => try testing.expect(u.role == .empty),
            }
        },
        .list => |items| {
            try testing.expectEqual(items.len, y.list.len);
            for (items, y.list) |p, q| try sameArgSide(p, q);
        },
        .i => |v| try testing.expectEqual(v, y.i),
        .f => |v| try testing.expectEqual(v, y.f),
        .b => |v| try testing.expectEqual(v, y.b),
        .none => try testing.expect(y == .none),
    }
}

test "overlapped ingest: the serial ingest's launches, its scratch roles its own" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    for ([_]i64{ 1, 5, 16 }) |n| for ([_]i64{ 0, 3 }) |skip| {
        const serial = try emit.emitIngestAt(a, &fx.cfg, &fx.w, fx.o, n, skip, .{ .ring_slots = 4, .slot = 2 });
        const side = try emit.sideRoles(a, serial);
        try testing.expectEqual(serial.len, side.len);
        var renamed: usize = 0;
        for (serial, side) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.triton, d.triton);
            try testing.expectEqual(c.grid, d.grid);
            // Runner.windowOn's contract: no branch, deferral or glue marks in a drafter program
            try testing.expect(!c.side and !c.fork and !c.join and !c.defer_side and !c.defer_join and !c.glue);
            try testing.expectEqual(c.args.len, d.args.len);
            for (c.args, d.args) |x, y| {
                try testing.expectEqualStrings(x.name, y.name);
                try sameArgSide(x.arg, y.arg);
                if (x.arg == .t and x.arg.t.role == .buf and emit.sideRole(x.arg.t.role.buf)) renamed += 1;
            }
        }
        try testing.expect(renamed > 0);
    };
}

fn rolesInto(a: std.mem.Allocator, set: *std.StringHashMapUnmanaged(void), x: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| if (t.role == .buf) try set.put(a, t.role.buf, {}),
        .list => |items| for (items) |y| try rolesInto(a, set, y),
        else => {},
    }
}

fn rolesOf(a: std.mem.Allocator, cs: []const calls.Call) !std.StringHashMapUnmanaged(void) {
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (cs) |c| for (c.args) |x| try rolesInto(a, &set, x.arg);
    return set;
}

test "overlapped ingest: a served row window's taps mark, and nothing the window touches but the taps and the rope table" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const cfg = &fx.cfg;
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    // the served decode shape: R1, row mode in the paged pool, taps, the branches and the deferred mHC coefficients
    var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .branches = true, .mhc_defer = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    const s: graphs.Settings = .{};
    const ing = try emit.sideRoles(a, try emit.emitIngestAt(a, cfg, &fx.w, fx.o, 16, 5, .{ .ring_slots = 4, .slot = 3 }));
    const side = try rolesOf(a, ing);
    for ([_]i64{ 5, 16, 20, 24 }) |n| {
        const ctx = graphs.bucketOf(300_000, s.bucket, s.grow);
        const one = try block.emit(a, cfg, &fx.w, o, &backbone, n, graphs.emitStart(@intCast(n), ctx, s, @intCast(limit)), true);
        const cs = try rowmode.transform(a, one, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = o.pool.?.pts });
        const m = run.lastTaps(cs) orelse return error.TestUnexpectedResult;
        // the mark sits past the last DSpark target's boundary (layer 39's), before the last layer's MoE and the head
        try testing.expect(m + 1 < cs.len);
        const after = try rolesOf(a, cs[m + 1 ..]);
        try testing.expect(!after.contains("w.taps"));
        const whole = try rolesOf(a, cs);
        var it = side.keyIterator();
        while (it.next()) |k| {
            const shared = std.mem.eql(u8, k.*, "w.taps") or std.mem.startsWith(u8, k.*, "s.rope.");
            if (whole.contains(k.*) and !shared) {
                std.debug.print("{d} rows: the side ingest's role {s} is the window's too\n", .{ n, k.* });
                return error.TestUnexpectedResult;
            }
            if (after.contains(k.*) and !std.mem.startsWith(u8, k.*, "s.rope.")) return error.TestUnexpectedResult;
        }
    }
}

test "device chain: emitChain is Python's captured _chain launch (names, dtypes, shapes, weights, constexprs)" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const cap = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/chain-op-cap16.json"), .{});
    const want = cap.object.get("args").?.array.items;
    const cs = try emit.emitChain(a, &fx.cfg, &fx.w, fx.o, fx.cfg.dspark_block, 1, 128, true);
    try testing.expectEqual(@as(usize, 1), cs.len);
    const c = cs[0];
    try testing.expectEqualStrings("_chain", c.name);
    try testing.expect(c.triton);
    const g = cap.object.get("grid").?.array.items;
    for (c.grid, g) |x, y| try testing.expectEqual(y.integer, x);
    try testing.expectEqual(want.len, c.args.len);
    for (c.args, want) |x, w| {
        const wo = w.object;
        try testing.expectEqualStrings(wo.get("name").?.string, x.name);
        const kind = wo.get("t").?.string;
        if (std.mem.eql(u8, kind, "tensor")) {
            const t = x.arg.t;
            const dt = wo.get("dtype").?.string;
            const ours: []const u8 = switch (t.dt) {
                .i32 => "int32",
                .i64 => "int64",
                .f32 => "float32",
                .f64 => "float64",
                .bf16 => "bfloat16",
                else => "?",
            };
            try testing.expectEqualStrings(dt, ours);
            for (wo.get("shape").?.array.items, t.shape) |p, q| try testing.expectEqual(p.integer, q);
            for (wo.get("stride").?.array.items, t.stride) |p, q| try testing.expectEqual(p.integer, q);
            if (wo.get("weight")) |wn| try testing.expectEqualStrings(wn.string, t.role.weight) else try testing.expect(t.role == .buf);
        } else if (std.mem.eql(u8, kind, "int")) {
            try testing.expectEqual(wo.get("v").?.integer, x.arg.i);
        } else if (std.mem.eql(u8, kind, "bool")) {
            try testing.expectEqual(wo.get("v").?.bool, x.arg.b);
        } else return error.TestUnexpectedResult;
    }
    // without the confidence head HID / CW / CONF are cval's tensor, D 1024
    const nc = (try emit.emitChain(a, &fx.cfg, &fx.w, fx.o, fx.cfg.dspark_block, 3, 128, false))[0];
    for (nc.args) |x| if (std.mem.eql(u8, x.name, "HID") or std.mem.eql(u8, x.name, "CW") or std.mem.eql(u8, x.name, "CONF"))
        try testing.expectEqualStrings(emit.chain_roles.cval, x.arg.t.role.buf);
    try testing.expectEqual(@as(i64, 3), nc.grid[0]);
}

test "device chain: ds_out's strided copies lay the gathered candidates out as the host's ungather" {
    const W = 2;
    const k = 4;
    const at = 3;
    const R = 7;
    const rows_max = at + R + 2;
    const c = W * k;
    // the gathered buffer: regions [W][rows][2 k] at each round row; ids as int bits after the values
    var gathered: [rows_max * 2 * k * W]f32 = undefined;
    for (&gathered, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i)) * 3 - 500);
    const region = gathered[at * 2 * k * W ..][0 .. R * 2 * k * W];
    for (0..W) |w| for (0..R) |i| for (0..k) |j| {
        region[(w * R + i) * 2 * k + k + j] = @bitCast(@as(i32, @intCast(1000 * w + 10 * i + j)));
    };
    var cand_want: [R * c]i32 = undefined;
    var cval_want: [R * c]f32 = undefined;
    rows.ungather(region, W, R, k, &cand_want, &cval_want);
    var cand: [rows_max * c]u32 = @splat(0xAAAAAAAA);
    var cval: [rows_max * c]u32 = @splat(0xAAAAAAAA);
    const src = std.mem.sliceAsBytes(&gathered);
    var cs: [8]dsd.Copy2D = undefined;
    for (dsd.reorder(W, R, k, at, &cs)) |cp| {
        const dst = std.mem.sliceAsBytes(if (cp.ids) &cand else &cval);
        for (0..cp.height) |h| @memcpy(dst[cp.dst + h * cp.dst_pitch ..][0..cp.width], src[cp.src + h * cp.src_pitch ..][0..cp.width]);
    }
    for (cand[at * c ..][0 .. R * c], cand_want) |x, y| try testing.expectEqual(@as(u32, @bitCast(y)), x);
    for (cval[at * c ..][0 .. R * c], cval_want) |x, y| try testing.expectEqual(@as(u32, @bitCast(y)), x);
    // nothing outside the region's rows
    for (cand[0 .. at * c]) |x| try testing.expectEqual(@as(u32, 0xAAAAAAAA), x);
    for (cval[(at + R) * c ..]) |x| try testing.expectEqual(@as(u32, 0xAAAAAAAA), x);
}

test "device chain: the staged inputs are Python's (sampling_params: ipar, anchor, fpar)" {
    var host: [4 * 48 + 2 * 4 * 5 * 4]u8 align(8) = undefined;
    const greedy = dsk.Params.of(null, 99);
    const smp = dsk.Params.of(.{ .seed = (1 << 63) | 77, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0.05 }, 1234);
    const got = dsd.Chain.stageArgs(&host, 4, &.{ 11, 22 }, &.{ greedy, smp });
    const ipar: []const i64 = @alignCast(std.mem.bytesAsSlice(i64, got.ipar));
    const anchor: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, got.anchor));
    const fpar: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, got.fpar));
    // greedy: [0, p + 1, 0, 0], [1.0, 1.0, 0.0]; sampled: [seed & (2^63 - 1), p + 1, top_k, 1], [T, top_p, min_p]
    try testing.expectEqualSlices(i64, &.{ 0, 100, 0, 0, 77, 1235, 20, 1 }, ipar);
    try testing.expectEqualSlices(i32, &.{ 11, 22 }, anchor);
    try testing.expectEqualSlices(f32, &.{ 1.0, 1.0, 0.0, 0.7, 0.95, 0.05 }, fpar);
}

// -- TF_DSV41_DRAFT_BATCH_PROJ ------------------------------------------------------------------------------------

/// `y` is `x` but for "w.ds.mx" tensors (lists included: the fused wkv's inputs), read `by` bytes further.
fn sameMoved(x: calls.Arg, y: calls.Arg, by: i64, moved: *usize) !void {
    switch (x) {
        .t => |t| if (t.role == .buf and std.mem.eql(u8, t.role.buf, "w.ds.mx")) {
            try testing.expectEqual(t.offset + by, y.t.offset);
            try testing.expectEqualSlices(i64, t.shape, y.t.shape);
            try testing.expectEqualSlices(i64, t.stride, y.t.stride);
            try testing.expect(@mod(y.t.offset, 16) == 0); // the slot's rows keep 16-byte alignment
            moved.* += 1;
        } else try testing.expect(equalArgs(x, y)),
        .list => |items| {
            try testing.expectEqual(items.len, y.list.len);
            for (items, y.list) |p, q| try sameMoved(p, q, by, moved);
        },
        else => try testing.expect(equalArgs(x, y)),
    }
}

test "batched main_proj: the projection then the blocks is the whole ingest; a slot's blocks read its rows of w.ds.mx" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const D: i64 = fx.cfg.hidden;
    for ([_]i64{ 1, 5, 16 }) |n| for ([_]i64{ 0, 7 }) |skip| {
        const sl: emit.Slots = .{ .ring_slots = 4, .slot = 1 };
        const whole = try emit.emitIngestAt(a, &fx.cfg, &fx.w, fx.o, n, skip, sl);
        const proj = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, n, skip, "w.taps");
        const blocks0 = try emit.emitIngestBlocks(a, &fx.cfg, &fx.w, fx.o, n, 0, sl);
        try identical(whole, try std.mem.concat(a, calls.Call, &.{ proj, blocks0 }));
        // at row r of a batched projection: the same calls, "w.ds.mx" read r rows further, nothing else moved
        const r: i64 = 11;
        const at = try emit.emitIngestBlocks(a, &fx.cfg, &fx.w, fx.o, n, r, sl);
        try testing.expectEqual(blocks0.len, at.len);
        var moved: usize = 0;
        for (blocks0, at) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.grid, d.grid);
            for (c.args, d.args) |x, y| try sameMoved(x.arg, y.arg, r * D * 2, &moved);
        }
        try testing.expect(moved > 0);
    };
}

test "batched main_proj: the projection's launches do not depend on its rows but for the row count" {
    // linear.cu's row bits depend on the row alone given (K, N, SK, WK): batching rows keeps every row's bits
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const one = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, 1, 0, "w.taps");
    for ([_]i64{ 2, 5, 16, 17, 20, 33, 64 }) |n| {
        const many = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, n, 3, "w.taps");
        try testing.expectEqual(one.len, many.len);
        for (one, many) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.triton, d.triton);
            for (c.args, d.args) |x, y| switch (x.arg) {
                .i, .f, .b, .none => try testing.expect(equalArgs(x.arg, y.arg)),
                .t => |t| {
                    try testing.expect(t.dt == y.arg.t.dt);
                    try testing.expectEqualSlices(i64, t.stride, y.arg.t.stride);
                    // rows are the leading dimension; the rest (K, N, the weights) are the same
                    if (t.shape.len > 1) try testing.expectEqualSlices(i64, t.shape[1..], y.arg.t.shape[1..]);
                },
                else => {},
            };
        }
    }
}

test "KV norm/store fuses DSpark ingest and multislot passes without touching other norms" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a);
    const off: block.Options = .{ .taps = true };
    var on = off;
    on.kv_norm_store = true;
    for ([_]i64{ 1, 16, 24, 64 }) |n| {
        const plain = try emit.emitIngest(a, &cfg, &w, off, n, 0);
        const fused = try emit.emitIngest(a, &cfg, &w, on, n, 0);
        try @import("kv_glue_emit_test.zig").compare(plain, fused, cfg.mtp_layers);
    }
    for ([_]i64{ 1, 2, 3, 4 }) |slots| {
        const plain = try emit.emitPassSlots(a, &cfg, &w, off, cfg.dspark_block, slots, slots);
        const fused = try emit.emitPassSlots(a, &cfg, &w, on, cfg.dspark_block, slots, slots);
        try @import("kv_glue_emit_test.zig").compare(plain, fused, cfg.mtp_layers);
        for (fused) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_kv_glue_v1.norm_store")) {
            try testing.expect(c.args[14].arg.b);
            try testing.expectEqual(@as(i64, slots * cfg.dspark_block), c.args[5].arg.t.shape[0]);
            try testing.expectEqual(@as(i64, slots * emit.ringRows(&cfg)), c.args[3].arg.t.shape[0]);
        };
    }
}
