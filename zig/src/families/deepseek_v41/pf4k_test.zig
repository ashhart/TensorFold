//! Host tests of 4,096-row prefill segments (TF_DSV41_PF_4K): the knob's rules and the emitter's 4K program.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

const ok = [2][]const u8{ "TF_DSV41_PF_4K_MEMORY_OK", "1" };
const on = [2][]const u8{ "TF_DSV41_PF_4K", "1" };

test "TF_DSV41_PF_4K: 0 / 1 only, refused without TF_DSV41_PF_4K_MEMORY_OK=1, 4,096-row segments (an explicit other size refused)" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.pf4k(&Fake.get));
    try testing.expectEqual(@as(?u32, null), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_4K", "0" }};
    try testing.expect(!try pk.pf4k(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_4K", "2" }};
    try testing.expectError(error.BadKnob, pk.pf4k(&Fake.get));
    Fake.pairs = &.{on};
    try testing.expectError(error.BadKnob, pk.pf4k(&Fake.get));
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on, .{ "TF_DSV41_PF_4K_MEMORY_OK", "0" } };
    try testing.expectError(error.BadKnob, pk.pf4k(&Fake.get));
    Fake.pairs = &.{ on, ok };
    try testing.expect(try pk.pf4k(&Fake.get));
    try testing.expectEqual(@as(?u32, 4096), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on, ok, .{ "TF_DSV41_PREFILL_CHUNK", "4096" }, .{ "TF_DSV41_PREFILL_ROWS", "4096" } };
    try testing.expectEqual(@as(?u32, 4096), try pk.prefillRows(&Fake.get));
    // the default words (2,048 both) with 4K on: refused, not 2,048-row segments under a 4K label
    Fake.pairs = &.{ on, ok, .{ "TF_DSV41_PREFILL_CHUNK", "2048" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on, ok, .{ "TF_DSV41_PREFILL_ROWS", "2048" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on, ok, .{ "TF_DSV41_PREFILL_ROWS", "4112" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    // without 4K the 2,048 limit stands
    Fake.pairs = &.{.{ "TF_DSV41_PREFILL_CHUNK", "4096" }};
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

fn isRoute(c: calls.Call) bool {
    return std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route");
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

/// `big`'s calls are `small`'s in order, with each route launch of `small` a run of `routes` route launches in `big`.
fn sameCalls(small: []const calls.Call, big: []const calls.Call, routes: usize) !void {
    var j: usize = 0;
    for (small) |c| {
        const k: usize = if (isRoute(c)) routes else 1;
        for (0..k) |_| {
            try testing.expect(j < big.len);
            const d = big[j];
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.glue, d.glue);
            try testing.expectEqual(c.args.len, d.args.len);
            for (c.args, d.args) |x, y| {
                try testing.expectEqual(std.meta.activeTag(x.arg), std.meta.activeTag(y.arg));
                if (x.arg == .t) {
                    try testing.expectEqualStrings(roleOf(x.arg), roleOf(y.arg));
                    try testing.expectEqual(x.arg.t.dt, y.arg.t.dt);
                }
            }
            j += 1;
        }
    }
    try testing.expectEqual(big.len, j);
}

test "a 4,096-row segment under TF_DSV41_PF_4K: the 2K program over twice the rows, the router in two CHUNK launches, x3gm one block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 }; // Engram, a ratio-2 kv source (compressor + indexer), a plain layer
    const D: i64 = cfg.hidden;
    const two: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    const four: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .pf4k = true, .prefill_rows = 4096 };
    // off: past 2,048 rows refused, whatever prefill_rows says (x3gm's rows would halve)
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, two, &layers, 4096, 0, true));
    var big_rows = two;
    big_rows.prefill_rows = 4096;
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, big_rows, &layers, 4096, 0, true));
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, four, &layers, 4112, 0, true));
    for ([_]i64{ 0, 6144 }) |start| {
        // no head (a prompt segment's logits: the head is 128 rows a launch, its own count at any rows)
        const s = try bp.emitPrefill(aa, &cfg, &w, four, &layers, 2048, start, false);
        const b = try bp.emitPrefill(aa, &cfg, &w, four, &layers, 4096, start, false);
        try sameCalls(s, b, 2);
        const moes = count(b, "tf_dsv41_x3gm_v1.rot");
        try testing.expectEqual(@as(usize, layers.len), moes);
        // x3gm: one plan, one gateup2, one down2 a MoE call, over every pair of the segment
        try testing.expectEqual(moes, count(b, "tf_dsv41_x3gm_v1.gateup2"));
        try testing.expectEqual(moes, count(b, "tf_dsv41_x3gm_v1.down2"));
        try testing.expectEqual(moes, count(b, "glue.gm_plan"));
        var routes: usize = 0;
        for (b) |c| {
            if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot")) {
                try testing.expectEqual(@as(i64, 4096 * 6), c.args[9].arg.i); // P
                try testing.expectEqual(@as(i64, 4096 * 6), c.args[5].arg.t.shape[0]); // X: [P, D]
            }
            if (!isRoute(c)) continue;
            // the halves: rows [0, 2,048) then [2,048, 4,096) of x / pick / wts / logits, prefill's tile and config
            const half: i64 = @intCast(routes % 2);
            const x = c.args[0].arg.t;
            try testing.expectEqual(@as(i64, 2048), x.shape[0]);
            try testing.expectEqual(half * 2048 * D * 2, x.offset);
            try testing.expectEqualStrings("w.out", roleOf(c.args[0].arg));
            for ([_]usize{ 3, 4, 5 }) |q| {
                const t = c.args[q].arg.t;
                try testing.expectEqual(@as(i64, 2048), t.shape[0]);
                try testing.expectEqual(half * 2048 * t.shape[1] * 4, t.offset);
            }
            try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
            try testing.expectEqual(@as(i64, 8), c.args[11].arg.i);
            try testing.expectEqual(@as(i64, 2), c.args[12].arg.i);
            routes += 1;
        }
        try testing.expectEqual(2 * moes, routes);
    }
    // a ragged last segment: the second launch's 8 rows at config(E, 8) = (8, 1), row tile 8 (use_narrow: 2,056 rows)
    const r = try bp.emitPrefill(aa, &cfg, &w, four, &layers, 2056, 0, false);
    var k: usize = 0;
    for (r) |c| if (isRoute(c)) {
        const second = k % 2 == 1;
        try testing.expectEqual(@as(i64, if (second) 8 else 2048), c.args[0].arg.t.shape[0]);
        try testing.expectEqual(@as(i64, if (second) 8 else 16), c.args[10].arg.i);
        try testing.expectEqual(@as(i64, if (second) 1 else 2), c.args[12].arg.i);
        k += 1;
    };
    try testing.expectEqual(2 * layers.len, k);
    // CED's encoder pass and a two-segment piece run at 4,096 rows
    var enc: [21]u32 = undefined; // layers 0 .. 20: 20 is the decoder's first (its site and compressor)
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    _ = try bp.emitReplay(aa, &cfg, &w, four, &enc, 4096, 0, .encoder);
    const sp = [_]bp.Span{ .{ .slot = 0, .start = 0, .n = 4096 - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
    _ = try bp.emitMulti(aa, &cfg, &w, four, &enc, &sp, .encoder);
}

test "TF_DSV41_PF_4K leaves a segment of 2,048 rows or fewer as it was, launch for launch, but for the staging ring" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const two: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var four = two;
    four.pf4k = true;
    four.prefill_rows = 4096;
    for ([_]i64{ 2048, 1024, 300, 17, 12 }) |n| {
        const a = try bp.emitPrefill(aa, &cfg, &w, two, &layers, n, 4000, true);
        const b = try bp.emitPrefill(aa, &cfg, &w, four, &layers, n, 4000, true);
        try testing.expectEqual(a.len, b.len);
        for (a, b) |x, y| {
            try testing.expectEqualStrings(x.name, y.name);
            try testing.expectEqual(x.grid, y.grid);
            try testing.expectEqual(x.args.len, y.args.len);
            for (x.args, y.args) |p, q| {
                // the staging ring's size: ceilPow2(prefill_rows + 127), its roles and its row count
                if (p.arg == .t and std.mem.startsWith(u8, roleOf(p.arg), "s.stage.")) continue;
                if (p.arg == .i and p.arg.i == 4096 and q.arg == .i and q.arg.i == 8192) continue;
                if (!(std.meta.eql(p.arg, q.arg) or argEql(p.arg, q.arg))) {
                    std.debug.print("{s} arg {s}: {any} vs {any}\n", .{ x.name, p.name, p.arg, q.arg });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

/// Two arguments with the same contents (tensors by role, dtype, shape, strides and offset; lists item by item).
fn argEql(p: calls.Arg, q: calls.Arg) bool {
    if (std.meta.activeTag(p) != std.meta.activeTag(q)) return false;
    return switch (p) {
        .t, .opaque_table => |t| blk: {
            const u = if (q == .t) q.t else q.opaque_table;
            break :blk std.mem.eql(u8, roleOf(.{ .t = t }), roleOf(.{ .t = u })) and t.dt == u.dt and t.offset == u.offset and
                std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .list => |xs| blk: {
            if (xs.len != q.list.len) break :blk false;
            for (xs, q.list) |x, y| if (!argEql(x, y)) break :blk false;
            break :blk true;
        },
        .i => |v| v == q.i,
        .f => |v| v == q.f,
        .b => |v| v == q.b,
        .none => true,
    };
}

const buffers = @import("buffers.zig");
const fp = @import("forward_prefill.zig");
const branches = @import("branches.zig");

test "TF_DSV41_PF_4K adds no byte to the forward's buffer plan: the 4K program's own roles are its workspace's, the rest the 2K plan's at their sizes" {
    // the forward's options under 4K are its 2K ones (the knob apply step leaves them); 4K segments use k4Options
    try planWithin(.{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20, .taps = true });
}

test "TF_DSV41_PF_4K + TF_DSV41_PF_OVERLAP (+ _SITE): the pieced 4K program stays in its workspace, its streams share only the partials (and the gathered buffers)" {
    for ([_]i64{ 2, 4 }) |k| for ([_]bool{ false, true }) |site| try planWithin(.{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20, .taps = true, .branches = true, .pf_overlap = k, .pf_overlap_site = site });
}

fn planWithin(base: block.Options) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const o4 = fp.k4Options(base);
    try testing.expectEqual(@as(i64, 4096), o4.prefill_rows);
    try testing.expect(o4.pf4k and !base.pf4k and base.prefill_rows == 2048);
    var p2: buffers.Plan = .{ .a = aa };
    var p4: buffers.Plan = .{ .a = aa };
    for ([_]i64{ 0, (1 << 20) - 4096 - 1 }) |start| {
        try p2.add(try bp.emitReplay(aa, &cfg, &w, base, &enc, 2048, start + 2048, .encoder));
        const c4 = try bp.ownCalls(aa, try bp.emitReplay(aa, &cfg, &w, o4, &enc, 4096, start, .encoder), fp.K4.tag);
        try p4.add(c4);
        if (base.pf_overlap > 1) {
            try testing.expect(count(c4, "glue.exchange_rows") > 0);
            // the pieces' all-gather scratch is the workspace's (priced with it)
            var xr = false;
            for (c4) |c| if (std.mem.eql(u8, c.name, "glue.exchange_rows")) {
                xr = true;
                try testing.expectEqualStrings("L.k4~xrows", roleOf(c.args[3].arg));
            };
            try testing.expect(xr);
            const sh = try branches.shared(aa, c4);
            try testing.expectEqual(@as(usize, if (base.pf_overlap_site) 4 else 2), sh.len);
            // the partials (and gathered buffers the site pieces read) both streams touch are the workspace's
            for (sh) |r| try testing.expect(std.mem.eql(u8, r, "L.k4~part") or std.mem.eql(u8, r, "L.k4~moe16") or
                (base.pf_overlap_site and (std.mem.eql(u8, r, "L.k4~recv.attn") or std.mem.eql(u8, r, "L.k4~recv.moe"))));
        }
    }
    var own: u64 = 0;
    for (p4.sizes.keys(), p4.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, fp.K4.tag) != null) {
            own += v;
            continue;
        }
        // shared with the forward: the slot's state (pool, rings, carries, keys, stash) and constant tables
        const b = p2.sizes.get(k) orelse {
            std.debug.print("the 4K program reads a role the 2K plan does not hold: {s}\n", .{k});
            return error.TestUnexpectedResult;
        };
        if (v > b) std.debug.print("the 4K program needs {s} at {d} bytes, the 2K plan holds {d}\n", .{ k, v, b });
        try testing.expect(v <= b);
        for (bp.own_scratch) |sc| try testing.expect(!std.mem.startsWith(u8, k, sc));
    }
    for (p2.sizes.keys()) |k| try testing.expect(std.mem.indexOf(u8, k, fp.K4.tag) == null);
    // the workspace: the 4K segment's scratch and window roles (x3gm's z at 24,576 pairs, the 8,192-row staging ring)
    try testing.expect(p4.sizes.get("s.k4~ex.z").? >= 4096 * 6 * 5120 * 4);
    try testing.expect(p4.sizes.contains("s.k4~stage.v") or p4.sizes.contains("s.k4~stage.s"));
    try testing.expect(own > 1 << 30);
    if (base.pf_overlap == 0) try shared4k(aa, &p2, &p4, own);
}

/// TF_DSV41_PF_4K alone: every role the 4K program names is the workspace's (no larger) or the 2K plan's (no smaller).
fn shared4k(aa: std.mem.Allocator, p2: *const buffers.Plan, p4: *const buffers.Plan, own: u64) !void {
    var wp: buffers.Plan = .{ .a = aa };
    for (p4.sizes.keys(), p4.sizes.values()) |k, v| try wp.sizes.put(aa, k, v);
    var keep: bp.Keep = .empty;
    try fp.k4Trim(aa, &wp, p2, &keep);
    for ([_][]const u8{ "L.ix.stream", "L.ix.keys", "s.pf.w" }) |r| try testing.expect(keep.contains(r));
    try testing.expect(!keep.contains("s.ex.z") and !keep.contains("L.q")); // twice the 2K size: the workspace's
    var trimmed: u64 = 0;
    for (wp.sizes.keys(), wp.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, fp.K4.tag) != null) trimmed += v;
    }
    try testing.expect(own >= trimmed + (400 << 20));
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const o4 = fp.k4Options(.{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20, .taps = true });
    for ([_]i64{ 0, (1 << 20) - 4096 - 1 }) |start| {
        var run: buffers.Plan = .{ .a = aa };
        try run.add(try bp.ownCallsKeep(aa, try bp.emitReplay(aa, &cfg, &w, o4, &enc, 4096, start, .encoder), fp.K4.tag, &keep));
        for (run.sizes.keys(), run.sizes.values()) |k, v| {
            if (std.mem.indexOf(u8, k, fp.K4.tag) != null) {
                try testing.expect(v <= wp.sizes.get(k).?);
            } else try testing.expect(v <= p2.sizes.get(k).?);
        }
    }
}

test "TF_DSV41_GM_GU2: 0 / 1, and a 4,096-row segment's gate / up at GU2 over its own 128-member plan, down unchanged" {
    Fake.pairs = &.{};
    try testing.expect(!try pk.gmGu2(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_GM_GU2", "1" }};
    try testing.expect(try pk.gmGu2(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_GM_GU2", "2" }};
    try testing.expectError(error.BadKnob, pk.gmGu2(&Fake.get));
    Fake.pairs = &.{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const four: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .pf4k = true, .prefill_rows = 4096 };
    var on_o = four;
    on_o.gm_gu2 = true;
    for ([_]i64{ 2048, 4096 }) |n| {
        const off = try bp.emitPrefill(aa, &cfg, &w, four, &layers, n, 0, false);
        const gu2 = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, n, 0, false);
        if (n < 4096) { // a 2K segment: the knob changes nothing
            try sameCalls(off, gu2, 1);
            continue;
        }
        // one more plan a MoE call (gate / up's, 128-member passes), every other call the same in order
        try testing.expectEqual(off.len + layers.len, gu2.len);
        var i: usize = 0;
        var gu: usize = 0;
        var dn: usize = 0;
        for (gu2, 0..) |c, j| {
            if (std.mem.eql(u8, c.name, "glue.gm_plan") and c.args[8].arg.i == 128) {
                // its roles: its own ("L.gm.h0.*"), the same picks
                try testing.expect(std.mem.startsWith(u8, roleOf(c.args[1].arg), "L.gm.h0."));
                try testing.expectEqualStrings(roleOf(off[i - 1].args[0].arg), roleOf(c.args[0].arg));
                continue;
            }
            try testing.expectEqualStrings(off[i].name, c.name);
            if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.gateup2")) {
                try testing.expectEqual(@as(i64, 2), c.args[18].arg.i); // GU2
                try testing.expect(std.mem.startsWith(u8, roleOf(c.args[5].arg), "L.gm.h0.")); // its plan
                try testing.expectEqualStrings("glue.gm_plan", gu2[j - 1].name);
                gu += 1;
            }
            if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.down2")) {
                try testing.expectEqual(@as(i64, 0), c.args[13].arg.i); // DN0
                try testing.expect(std.mem.startsWith(u8, roleOf(c.args[3].arg), "L.gm.g0.")); // the 64-member plan
                for (c.args, off[i].args) |x, y| try testing.expectEqualDeep(y.arg, x.arg);
                dn += 1;
            }
            i += 1;
        }
        try testing.expectEqual(off.len, i);
        try testing.expectEqual(layers.len, gu);
        try testing.expectEqual(layers.len, dn);
    }
}
