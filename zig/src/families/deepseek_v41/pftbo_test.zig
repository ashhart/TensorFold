//! Host tests of the prefill two-batch overlap (TF_DSV41_PF_TBO): readers, pair split, and emitTbo's two streams.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const branches = @import("branches.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const fp = @import("forward_prefill.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_TBO and _GM_SMS: 0 / 1; 0 or 8-48 SMs, 0 (every SM) by default; the pair split" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.pfTbo(&Fake.get));
    try testing.expectEqual(pk.tbo_gm_sms, try pk.pfTboGmSms(&Fake.get));
    Fake.pairs = &.{ .{ "TF_DSV41_PF_TBO", "1" }, .{ "TF_DSV41_PF_TBO_GM_SMS", "36" } };
    try testing.expect(try pk.pfTbo(&Fake.get));
    try testing.expectEqual(@as(u32, 36), try pk.pfTboGmSms(&Fake.get));
    Fake.pairs = &.{ .{ "TF_DSV41_PF_TBO", "2" }, .{ "TF_DSV41_PF_TBO_GM_SMS", "4" } };
    try testing.expectError(error.BadKnob, pk.pfTbo(&Fake.get));
    try testing.expectError(error.BadKnob, pk.pfTboGmSms(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_TBO_GM_SMS", "0" }};
    try testing.expectEqual(@as(u32, 0), try pk.pfTboGmSms(&Fake.get));
    // two whole segments; a shorter rest halved on the 16-row grid; under 2 x 512 rows one segment at a time
    try testing.expectEqual([2]usize{ 2048, 2048 }, fp.pairSplit(2048, 9000).?);
    try testing.expectEqual([2]usize{ 1504, 1504 }, fp.pairSplit(2048, 3008).?);
    try testing.expectEqual([2]usize{ 1504, 1505 }, fp.pairSplit(2048, 3009).?);
    try testing.expectEqual([2]usize{ 512, 512 }, fp.pairSplit(2048, 1024).?);
    try testing.expectEqual(@as(?[2]usize, null), fp.pairSplit(2048, 1023));
    try testing.expectEqual([2]usize{ 4096, 4096 }, fp.pairSplit(4096, 8192).?);
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn roleOf(x: calls.Arg) ?[]const u8 {
    return switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |b| b,
            else => null,
        },
        else => null,
    };
}

/// Call `x` of a program == call `y` of the pair with every role mapped by `map` (null: unchanged).
fn sameCall(a: std.mem.Allocator, x: calls.Call, y: calls.Call, b_side: bool) !void {
    try testing.expectEqualStrings(x.name, y.name);
    try testing.expectEqual(x.grid, y.grid);
    try testing.expectEqual(x.args.len, y.args.len);
    for (x.args, y.args) |p, q| {
        try testing.expectEqualStrings(p.name, q.name);
        try testing.expectEqual(std.meta.activeTag(p.arg), std.meta.activeTag(q.arg));
        if (roleOf(p.arg)) |r| {
            const want = if (b_side) try bp.tboRole(a, r) else r;
            try testing.expectEqualStrings(want, roleOf(q.arg).?);
            try testing.expectEqual(p.arg.t.offset, q.arg.t.offset);
            try testing.expectEqualSlices(i64, p.arg.t.shape, q.arg.t.shape);
        } else if (p.arg == .i) try testing.expectEqual(p.arg.i, q.arg.i);
    }
}

/// Constant tables both streams read (weight forms, pointer tables, rope, page tables): all a fork may share.
fn constant(r: []const u8) bool {
    if (std.mem.eql(u8, r, "s.pf.had") or std.mem.eql(u8, r, "s.kv.pt") or std.mem.eql(u8, r, "s.kv.ct")) return true;
    if (std.mem.startsWith(u8, r, "s.rope.")) return true;
    for ([_][]const u8{ ".lanes", ".T", ".fn16" }) |s| if (std.mem.endsWith(u8, r, s)) return true;
    for ([_][]const u8{ ".gm.", ".ex.tp_", ".pf." }) |s| if (std.mem.startsWith(u8, r, "s.L") and std.mem.indexOf(u8, r, s) != null) return true;
    return false;
}

/// A kv source's compressed rows: A's later layers read them up to A's positions while B appends past A's last.
fn sourceRows(cfg: *const Config, r: []const u8) bool {
    const pre = if (std.mem.startsWith(u8, r, "s.kv.comp.L")) "s.kv.comp.L" else if (std.mem.startsWith(u8, r, "s.L")) "s.L" else return false;
    const rest = r[pre.len..];
    const end = std.mem.indexOfNone(u8, rest, "0123456789") orelse rest.len;
    const L = std.fmt.parseInt(u32, rest[0..end], 10) catch return false;
    if (!cfg.isKvSource(L)) return false;
    return pre.len == "s.kv.comp.L".len or std.mem.eql(u8, rest[end..], ".comp.v") or std.mem.eql(u8, rest[end..], ".comp.s");
}

test "TF_DSV41_PF_TBO on the emitter: A's program on main, B's renamed on the side half a layer behind, forks, carries, joins" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined; // CED's encoder: layers 0 .. 20 (20's site and compressor)
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const o: block.Options = .{ .gm_v2 = .one, .branches = true, .pf_tbo = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    for ([_][3]i64{ .{ 2048, 2048, 0 }, .{ 1504, 1505, 6144 }, .{ 512, 512, 40000 } }) |cse| {
        const na = cse[0];
        const nb = cse[1];
        const start = cse[2];
        const pair = try bp.emitTbo(aa, &cfg, &w, o, &enc, na, nb, start);
        const pa = try bp.emitReplay(aa, &cfg, &w, o, &enc, na, start, .encoder);
        const pb = try bp.emitReplay(aa, &cfg, &w, o, &enc, nb, start + na, .encoder);
        // the streams' calls in order: A's program but its stash on main (+ carries, closing join), B's renamed on side
        var ia: usize = 0;
        var ib: usize = 0;
        var carries: usize = 0;
        var joins: usize = 0;
        var b_started = false;
        for (pair, 0..) |c, k| {
            if (std.mem.eql(u8, c.name, "glue.copy_rows")) {
                try testing.expect(!c.side);
                const src = c.args[0].arg.t;
                try testing.expectEqual((na - 1) * 2 * @as(i64, cfg.head_dim) * 4, src.offset);
                try testing.expect(std.mem.startsWith(u8, roleOf(c.args[1].arg).?, "s.L") and std.mem.endsWith(u8, roleOf(c.args[1].arg).?, ".carry"));
                carries += 1;
                continue;
            }
            if (std.mem.eql(u8, c.name, "glue.tbo_join")) {
                try testing.expect(c.join and k + 1 == pair.len);
                continue;
            }
            if (c.join) joins += 1;
            if (c.side) {
                if (!b_started) try testing.expect(c.fork);
                b_started = true;
                try sameCall(aa, pb[ib], c, true);
                ib += 1;
            } else {
                while (std.mem.eql(u8, pa[ia].name, "glue.ced_stash")) ia += 1;
                try sameCall(aa, pa[ia], c, false);
                ia += 1;
            }
            // no role of A's is B's scratch, every B role is renamed or the slot's state / a constant
            for (c.args) |x| if (roleOf(x.arg)) |r| {
                const renamed = std.mem.indexOf(u8, r, "b~") != null;
                try testing.expectEqual(renamed and true, renamed and c.side);
            };
        }
        while (ia < pa.len and std.mem.eql(u8, pa[ia].name, "glue.ced_stash")) ia += 1;
        try testing.expectEqual(pa.len, ia);
        try testing.expectEqual(pb.len, ib);
        // the ratio-2 kv sources among layers 0 .. 19 hand their carry over; A's Engram layer 14 part joins B
        var r2: usize = 0;
        for (enc) |L| r2 += @intFromBool(cfg.isKvSource(L) and cfg.compressRatio(L) == 2);
        try testing.expect(r2 > 0);
        try testing.expectEqual(r2, carries);
        try testing.expectEqual(@as(usize, 1), joins);
        // each carry is in the slot before B's same layer: the copy precedes the fork of B's next attention part
        for (pair, 0..) |c, k| if (std.mem.eql(u8, c.name, "glue.copy_rows")) {
            try testing.expect(pair[k + 1].side and pair[k + 1].fork);
        };
        // the streams share nothing inside a fork region but constants
        const sh = try branches.shared(aa, pair);
        var bad = false;
        for (sh) |r| {
            bad = bad or !(constant(r) or sourceRows(&cfg, r));
            if (!(constant(r) or sourceRows(&cfg, r))) std.debug.print("shared across the streams: {s}\n", .{r});
        }
        try testing.expect(!bad);
    }
    // refused: a micro-batch under 512 rows, no side stream, TBO off
    try testing.expectError(error.Unsupported, bp.emitTbo(aa, &cfg, &w, o, &enc, 496, 2048, 0));
    try testing.expectError(error.Unsupported, bp.emitTbo(aa, &cfg, &w, o, &enc, 2048, 2048, 1)); // an odd boundary
    var off = o;
    off.branches = false;
    try testing.expectError(error.Unsupported, bp.emitTbo(aa, &cfg, &w, off, &enc, 2048, 2048, 0));
    off = o;
    off.pf_tbo = false;
    try testing.expectError(error.Unsupported, bp.emitTbo(aa, &cfg, &w, off, &enc, 2048, 2048, 0));
}

const buffers = @import("buffers.zig");

test "a TBO pair adds no byte to the forward's plan: 2K pairs keep A's roles and put B's in the workspace, 4K pairs put both there" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const base: block.Options = .{ .gm_v2 = .one, .branches = true, .pf_tbo = true, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20, .taps = true };
    var p2: buffers.Plan = .{ .a = aa };
    for ([_]i64{ 0, (1 << 20) - 4096 - 1 }) |start| try p2.add(try bp.emitReplay(aa, &cfg, &w, base, &enc, 2048, start + 2048, .encoder));
    for ([_]bool{ false, true }) |k4| {
        const o = if (k4) fp.k4Options(base) else base;
        const seg: i64 = if (k4) 4096 else 2048;
        var pp: buffers.Plan = .{ .a = aa };
        const pair = try fp.wsPair(aa, try bp.emitTbo(aa, &cfg, &w, o, &enc, seg, seg, 8192), k4);
        try pp.add(pair);
        var own: u64 = 0;
        for (pp.sizes.keys(), pp.sizes.values()) |k, v| {
            if (std.mem.indexOf(u8, k, fp.K4.tag) != null) {
                own += v;
                continue;
            }
            try testing.expect(std.mem.indexOf(u8, k, "b~") == null); // every B role is the workspace's
            const b = p2.sizes.get(k) orelse {
                std.debug.print("pair (4K {}) reads a role the 2K plan does not hold: {s}\n", .{ k4, k });
                return error.TestUnexpectedResult;
            };
            if (v > b) std.debug.print("pair (4K {}) needs {s} at {d}, the 2K plan holds {d}\n", .{ k4, k, v, b });
            try testing.expect(v <= b);
            if (k4) for (bp.own_scratch) |sc| try testing.expect(!std.mem.startsWith(u8, k, sc));
        }
        // B's x3gm scratch in the workspace at its micro-batch's pairs; under 4K A's too
        try testing.expect(pp.sizes.get("s.k4~b~ex.z").? >= @as(u64, @intCast(seg * 6 * 5120 * 4)));
        try testing.expectEqual(k4, pp.sizes.contains("s.k4~ex.z"));
        try testing.expect(own > 1 << 29);
        // the same streams' structure as the plain pair (renaming moves no call)
        for (pair) |c| if (std.mem.eql(u8, c.name, "glue.copy_rows")) try testing.expect(!c.side);
    }
}

test "TF_DSV41_PF_TBO with TF_DSV41_PF_4K: 4,096-row micro-batches, each its 4K encoder program (route in two CHUNK launches, x3gm one block)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const o = fp.k4Options(.{ .gm_v2 = .one, .branches = true, .pf_tbo = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 });
    const pair = try bp.emitTbo(aa, &cfg, &w, o, &enc, 4096, 4096, 0);
    const pa = try bp.emitReplay(aa, &cfg, &w, o, &enc, 4096, 0, .encoder);
    const pb = try bp.emitReplay(aa, &cfg, &w, o, &enc, 4096, 4096, .encoder);
    var ia: usize = 0;
    var ib: usize = 0;
    for (pair) |c| {
        if (std.mem.eql(u8, c.name, "glue.copy_rows") or std.mem.eql(u8, c.name, "glue.tbo_join")) continue;
        if (c.side) {
            try sameCall(aa, pb[ib], c, true);
            ib += 1;
        } else {
            while (std.mem.eql(u8, pa[ia].name, "glue.ced_stash")) ia += 1;
            try sameCall(aa, pa[ia], c, false);
            ia += 1;
        }
    }
    try testing.expectEqual(pb.len, ib);
    var gu: usize = 0;
    var routes: usize = 0;
    for (pair) |c| {
        if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.gateup2")) gu += 1;
        if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) routes += 1;
    }
    try testing.expectEqual(routes, 2 * gu); // two CHUNK launches a MoE call, one gateup2
    // a 4K pair split from a short prompt rest
    try testing.expectEqual([2]usize{ 4096, 4096 }, fp.pairSplit(4096, 8192).?);
    try testing.expectEqual([2]usize{ 3008, 3008 }, fp.pairSplit(4096, 6016).?);
}
