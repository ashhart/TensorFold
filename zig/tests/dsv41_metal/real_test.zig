//! A REAL release layer against float64 at 1 and 4 rows: TF_DSV41_REAL (config + shard), TF_DSV41_REAL_LAYER (2).
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");
const check_moe = @import("check_moe.zig");

const testing = std.testing;
const limit = 1280;

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// A rig of the release's config with only `layer` loaded (no embed or head).
fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, layer: u32, zero_copy: bool) !*fixture.Rig {
    const r = try a.create(fixture.Rig);
    errdefer a.destroy(r);
    r.a = a;
    r.c = try host.config.Config.read(a, io, dir);
    r.device = try mtl.Device.init();
    r.queue = try r.device.queue();
    r.p = try pkg.pipelines.Pipelines.load(a, r.device, &r.c);
    var pack = try host.pack.Pack.open(a, io, dir);
    defer pack.deinit();
    r.map = if (zero_copy) try pkg.mapped.Map.open(a, r.device, &pack) else null;
    r.layers = try a.alloc(pkg.weights.Layer, 1);
    r.layers[0] = try pkg.weights.loadFrom(a, io, r.device, &pack, &r.c, layer, if (r.map) |*m| m else null);
    r.top = null;
    r.tables = try pkg.rope.Tables.init(r.device, &r.c, limit);
    r.s = try pkg.scratch.Scratch.init(r.device, &r.c, limit);
    r.slot = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    const sizes = [4]usize{ 16 * 4 * r.c.hidden * 2, 16 * 4 * r.c.hidden * 2, 16 * 16, 16 * 16 };
    for (&r.io, sizes) |*b, n| b.* = try r.device.buffer(n, mtl.ResourceOptions.shared);
    return r;
}

fn run(a: std.mem.Allocator, r: *fixture.Rig, p0: u32, rows: usize, seed: u64) !usize {
    const H = r.c.hidden;
    const w = &r.layers[0];
    fixture.fillPast(&r.c, &r.slot, p0, seed);
    fixture.fillInput(r, rows, seed + 1);
    const x = try a.dupe(u16, r.io[0].slice(u16, rows * 4 * H));
    defer a.free(x);
    const pre = try a.dupe(f32, r.io[2].slice(f32, rows * 4));
    defer a.free(pre);
    const carry = if (r.slot.regions[w.layer].carry) |cb| try a.dupe(f32, cb.slice(f32, 4 * r.c.head_dim)) else null;
    defer if (carry) |v| a.free(v);
    const ctx = r.ctx(&r.slot, p0, @intCast(rows));
    try r.s.writeRows(&r.c, p0, @intCast(rows));
    const pass = fixture.Pass.begin(r.queue);
    const s = r.streams();
    const pr = r.pres();
    try pkg.layer.encode(&ctx, pass.e, w, .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
    try pass.finish();
    var pb: probe.Probe = .{ .a = a, .r = r, .w = w, .rows = rows, .p0 = p0 };
    defer pb.deinit();
    try check_attn.check(&pb, .{ .x = x, .pre = pre, .carry = carry });
    try check_moe.check(&pb, pr[1], s[1]);
    std.debug.print("real layer {d}, {d} row(s) a launch from position {d}:\n", .{ w.layer, rows, p0 });
    return pb.report();
}

test "dsv41 metal REAL layer: every stage within its float64 bound at 1 and 4 rows" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const layer = if (env("TF_DSV41_REAL_LAYER")) |v| try std.fmt.parseInt(u32, v, 10) else 2;
    const a = std.heap.page_allocator;
    const r = open(a, testing.io, dir, layer, false) catch |e| {
        std.debug.print("FAIL dsv41 metal REAL layer {d} from {s}: {t}\n", .{ layer, dir, e });
        return e;
    };
    defer r.deinit();
    var bad: usize = 0;
    bad += try run(a, r, 1101, 1, 21);
    bad += try run(a, r, 1101, 4, 31);
    try testing.expectEqual(@as(usize, 0), bad);
}

/// The launch's output stream and pre mix after `run`, as bytes.
fn outputs(a: std.mem.Allocator, r: *fixture.Rig, rows: usize) ![]u8 {
    const x = r.io[1].contents()[0 .. rows * 4 * r.c.hidden * 2];
    const pre = r.io[3].contents()[0 .. rows * 16];
    return std.mem.concat(a, u8, &.{ x, pre });
}

test "dsv41 metal REAL layer over a mapped shard: float64 bounds hold and every output bit equals the copy" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const layer = if (env("TF_DSV41_REAL_LAYER")) |v| try std.fmt.parseInt(u32, v, 10) else 2;
    const a = std.heap.page_allocator;
    const copy = try open(a, testing.io, dir, layer, false);
    defer copy.deinit();
    const zero = try open(a, testing.io, dir, layer, true);
    defer zero.deinit();
    const st = zero.map.?.stats;
    std.debug.print("mapped: {d} shard(s) {d} B wired, {d} tensors ({d} B) in place, {d} refused ({d} B); the arena holds {d} B\n", .{ st.shards, st.wired, st.tensors, st.bytes, st.refused, st.refused_bytes, zero.layers[0].arena.length() });
    var bad: usize = 0;
    for ([_][3]u64{ .{ 1101, 1, 21 }, .{ 1101, 4, 31 } }) |q| {
        bad += try run(a, copy, @intCast(q[0]), @intCast(q[1]), q[2]);
        bad += try run(a, zero, @intCast(q[0]), @intCast(q[1]), q[2]);
        const x = try outputs(a, copy, @intCast(q[1]));
        defer a.free(x);
        const y = try outputs(a, zero, @intCast(q[1]));
        defer a.free(y);
        const same = std.mem.eql(u8, x, y);
        std.debug.print("{s} real layer {d}, {d} row(s): mapped output == copy output bit for bit ({d} bytes)\n", .{ if (same) "PASS" else "FAIL", layer, q[1], x.len });
        try testing.expect(same);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// One launch of the layer over each of `slots` with `per` rows a slot, committed and waited for, in microseconds.
fn timeLaunch(io: std.Io, r: *fixture.Rig, slots: []const pkg.kv.Slot, p0: u32, per: u32) !f64 {
    const w = &r.layers[0];
    var spans: [4]pkg.scratch.Span = undefined;
    var segs: [4]pkg.ctx.Seg = undefined;
    for (slots, 0..) |*sl, i| {
        spans[i] = .{ .p0 = p0, .n = per };
        segs[i] = .{ .slot = sl, .p0 = p0, .r0 = @as(u32, @intCast(i)) * per, .n = per };
    }
    const total: u32 = per * @as(u32, @intCast(slots.len));
    try r.s.writeSpans(&r.c, spans[0..slots.len]);
    var x = r.ctx(&slots[0], p0, total);
    x.segs = segs[0..slots.len];
    const t0 = std.Io.Clock.awake.now(io);
    const pass = fixture.Pass.begin(r.queue);
    const s = r.streams();
    const pr = r.pres();
    try pkg.layer.encode(&x, pass.e, w, .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
    try pass.finish();
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1000.0;
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

test "dsv41 metal REAL layer: decode time of 1 slot, 4 slots in one launch and 4 launches (TF_DSV41_REAL)" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const layer = if (env("TF_DSV41_REAL_LAYER")) |v| try std.fmt.parseInt(u32, v, 10) else 2;
    const a = std.heap.page_allocator;
    const r = try open(a, testing.io, dir, layer, false);
    defer r.deinit();
    var slots: [4]pkg.kv.Slot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
        fixture.fillPast(&r.c, sl, 1101, 40 + i);
    }
    defer for (&slots) |*sl| sl.deinit();
    fixture.fillInput(r, 16, 5);
    for ([_]u32{ 1, 4 }) |per| {
        var one: [15]f64 = undefined;
        var four: [15]f64 = undefined;
        var apart: [15]f64 = undefined;
        for (0..one.len + 2) |i| {
            const t1 = try timeLaunch(testing.io, r, slots[0..1], 1101, per);
            const t4 = try timeLaunch(testing.io, r, slots[0..4], 1101, per);
            var t = try timeLaunch(testing.io, r, slots[0..1], 1101, per);
            for (1..4) |k| t += try timeLaunch(testing.io, r, slots[k .. k + 1], 1101, per);
            if (i >= 2) {
                one[i - 2] = t1;
                four[i - 2] = t4;
                apart[i - 2] = t;
            }
        }
        std.debug.print("real layer {d}, {d} row(s) a slot at 1101: 1 slot {d:.0} us, 4 slots in one launch {d:.0} us, 4 launches {d:.0} us (median of {d})\n", .{ layer, per, median(&one), median(&four), median(&apart), one.len });
    }
}
