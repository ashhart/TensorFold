//! DeepSeek-V4.1's Metal decode forward: every stage within its float64 bound, rows at once == one at a time.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");
const check_moe = @import("check_moe.zig");
const check_fwd = @import("check_fwd.zig");

const testing = std.testing;
const fwd = pkg.forward;
const limit = 1280;
pub const look = 3;

fn copy(comptime T: type, a: std.mem.Allocator, r: probe.Ref, n: usize) ![]T {
    return a.dupe(T, @as([*]const T, @ptrCast(@alignCast(r.buf.contents() + r.off)))[0..n]);
}

/// The forward one stage group a pass (embed, each layer, head), every stage checked after its pass.
pub fn stagewise(a: std.mem.Allocator, f: *ff.Fwd, p0: u32, rows: usize, seed: u64, controls: bool) !usize {
    const r = f.r;
    const c = &r.c;
    const H = c.hidden;
    const m = f.model();
    const hist = try ff.ids(a, c, look + rows, seed);
    defer a.free(hist);
    try fwd.prepare(&f.b, c, hist[look..], hist[0..look], f.lookup());
    try r.s.writeRows(c, p0, @intCast(rows));
    const x = r.ctx(&r.slot, p0, @intCast(rows));
    var bad: usize = 0;
    {
        const pass = fixture.Pass.begin(r.queue);
        fwd.encodeEmbed(&x, pass.e, m, &f.b);
        try pass.finish();
        var pb: probe.Probe = .{ .a = a, .r = r, .w = &r.layers[0], .rows = rows, .p0 = p0 };
        defer pb.deinit();
        try check_fwd.embed(&pb, f, hist[look..]);
        bad += pb.report();
    }
    for (r.layers, 0..) |*w, i| {
        const s = i % 2;
        const xin = try copy(u16, a, f.b.streams[s], rows * 4 * H);
        defer a.free(xin);
        const pre = try copy(f32, a, f.b.pres[s], rows * 4);
        defer a.free(pre);
        const carry = if (r.slot.regions[i].carry) |cb| try a.dupe(f32, cb.slice(f32, 4 * c.head_dim)) else null;
        defer if (carry) |v| a.free(v);
        const pass = fixture.Pass.begin(r.queue);
        try fwd.encodeLayer(&x, pass.e, m, &f.b, i);
        try pass.finish();
        var pb: probe.Probe = .{ .a = a, .r = r, .w = w, .rows = rows, .p0 = p0 };
        defer pb.deinit();
        if (std.mem.indexOfScalar(u32, c.engram_layers.items(), w.layer)) |li| {
            if (controls) f.b.bytes(f.b.rows[li], 2)[1] ^= 0x01;
            try check_fwd.engram(&pb, f, li, hist, xin);
        }
        const xpost = try copy(u16, a, f.b.streams[s], rows * 4 * H);
        defer a.free(xpost);
        if (std.mem.indexOfScalar(u32, c.dspark_targets.items(), w.layer)) |t| try check_fwd.tap(&pb, xpost, t);
        try check_attn.check(&pb, .{ .x = xpost, .pre = pre, .carry = carry });
        try check_moe.check(&pb, f.b.pres[1 - s], f.b.streams[1 - s]);
        bad += if (controls) red(&pb, "engram rows (hash + table)") else pb.report();
    }
    const last = r.layers.len % 2;
    const xl = try copy(u16, a, f.b.streams[last], rows * 4 * H);
    defer a.free(xl);
    const pl = try copy(f32, a, f.b.pres[last], rows * 4);
    defer a.free(pl);
    const pass = fixture.Pass.begin(r.queue);
    fwd.encodeHead(&x, pass.e, m, &f.b);
    try pass.finish();
    var pb: probe.Probe = .{ .a = a, .r = r, .w = &r.layers[r.layers.len - 1], .rows = rows, .p0 = p0, .at = r.layers.len };
    defer pb.deinit();
    const hw = f.r.top.?.head.buf.contents() + f.r.top.?.head.off;
    if (controls) hw[1] ^= 0x01;
    try check_fwd.head(&pb, f, xl, pl);
    if (controls) hw[1] ^= 0x01;
    bad += if (controls) red(&pb, "head logits") else pb.report();
    return bad;
}

/// 1 when the named stage went red (a planted fault caught), printed either way.
fn red(pb: *const probe.Probe, name: []const u8) usize {
    for (pb.stages.items) |s| if (std.mem.eql(u8, s.name, name)) {
        std.debug.print("control {s} (layer {d}): {d} of {d} values red\n", .{ name, pb.at orelse pb.w.layer, s.bad, s.n });
        return @intFromBool(s.bad > 0);
    };
    return 0;
}

test "dsv41 metal forward: every stage of the tiny model within its float64 bound (Engram, taps, head)" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 51);
    defer f.deinit();
    fixture.fillPast(&f.r.c, &f.r.slot, 1101, 7);
    try testing.expectEqual(@as(usize, 0), try stagewise(a, f, 1101, 3, 13, false));
}

test "dsv41 metal forward: planted faults turn their stage checks red (an Engram row, the head)" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 52);
    defer f.deinit();
    fixture.fillPast(&f.r.c, &f.r.slot, 1101, 8);
    // two Engram layers' rows and the head: 3 planted faults
    const caught = try stagewise(a, f, 1101, 2, 14, true);
    std.debug.print("{s} dsv41 metal forward controls: {d} of 3 planted faults red\n", .{ if (caught == 3) "PASS" else "FAIL", caught });
    try testing.expectEqual(@as(usize, 3), caught);
}

/// Logits [n][vocab] and taps [n][targets][hidden] of n steps.
pub const Out = struct { logits: []f32, taps: []u16 };

/// The whole forward over `n` steps from p0 in launches of `chunk` rows on `slot`.
pub fn runChunks(a: std.mem.Allocator, f: *ff.Fwd, slot: *const pkg.kv.Slot, hist: []const u32, p0: u32, chunk: usize) !Out {
    const r = f.r;
    const c = &r.c;
    const n = hist.len - look;
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    const out: Out = .{ .logits = try a.alloc(f32, n * V), .taps = try a.alloc(u16, n * TH) };
    var at: usize = 0;
    while (at < n) : (at += chunk) {
        const k = @min(chunk, n - at);
        try fwd.prepare(&f.b, c, hist[look + at ..][0..k], ff.lookbackAt(hist, look + at, look), f.lookup());
        const p: u32 = p0 + @as(u32, @intCast(at));
        try r.s.writeRows(c, p, @intCast(k));
        const x = r.ctx(slot, p, @intCast(k));
        const pass = fixture.Pass.begin(r.queue);
        try fwd.encode(&x, pass.e, f.model(), &f.b);
        try pass.finish();
        @memcpy(out.logits[at * V ..][0 .. k * V], f.b.logitsOf(c, k));
        @memcpy(out.taps[at * TH ..][0 .. k * TH], slot.taps.?.slice(u16, k * TH));
    }
    return out;
}

fn reset(r: *fixture.Rig, to: *const pkg.kv.Slot, from: *const pkg.kv.Slot) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    to.copyFrom(cb, from);
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CommandBufferFailed;
}

fn stateDiff(x: *const pkg.kv.Slot, y: *const pkg.kv.Slot) usize {
    var n: usize = 0;
    for (x.regions, y.regions) |p, q| {
        const pairs = [_][2]?mtl.Buffer{ .{ p.window, q.window }, .{ p.pool, q.pool }, .{ p.keys, q.keys }, .{ p.carry, q.carry } };
        for (pairs) |pr| if (pr[0]) |bx| {
            const len = bx.length();
            for (bx.contents()[0..len], pr[1].?.contents()[0..len]) |s, t| n += @intFromBool(s != t);
        };
    }
    return n;
}

/// One stream's `n` steps through the whole forward in launches of `chunk` rows and one at a time, bit for bit.
fn compare(a: std.mem.Allocator, f: *ff.Fwd, pristine: *const pkg.kv.Slot, p0: u32, n: usize, chunk: usize, seed: u64) !usize {
    const r = f.r;
    var one = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer one.deinit();
    var many = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer many.deinit();
    try reset(r, &one, pristine);
    try reset(r, &many, pristine);
    const hist = try ff.ids(a, &r.c, look + n, seed);
    defer a.free(hist);
    const o1 = try runChunks(a, f, &one, hist, p0, 1);
    defer a.free(o1.logits);
    defer a.free(o1.taps);
    const on = try runChunks(a, f, &many, hist, p0, chunk);
    defer a.free(on.logits);
    defer a.free(on.taps);
    var bad: usize = 0;
    for (o1.logits, on.logits) |s, t| bad += @intFromBool(@as(u32, @bitCast(s)) != @as(u32, @bitCast(t)));
    var tb: usize = 0;
    for (o1.taps, on.taps) |s, t| tb += @intFromBool(s != t);
    const st = stateDiff(&one, &many);
    const word = if (bad + tb + st == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal forward ({d} layers): {d} steps from {d}, {d} rows a launch vs one: {d} logits, {d} tap values, {d} state bytes differ\n", .{ word, r.layers.len, n, p0, chunk, bad, tb, st });
    return bad + tb + st;
}

test "dsv41 metal forward: R rows in one launch equal the same rows one at a time, R = 1..16" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 53);
    defer f.deinit();
    var pristine = try pkg.kv.Slot.init(a, f.r.device, &f.r.c, limit);
    defer pristine.deinit();
    fixture.fillPast(&f.r.c, &pristine, 1101, 9);
    var bad: usize = 0;
    for (1..17) |rows| bad += try compare(a, f, &pristine, 1101, rows, rows, 200 + rows);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal forward: 8 decode steps, 4 rows a launch equal one row a launch" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 54);
    defer f.deinit();
    var pristine = try pkg.kv.Slot.init(a, f.r.device, &f.r.c, limit);
    defer pristine.deinit();
    fixture.fillPast(&f.r.c, &pristine, 1104, 10);
    try testing.expectEqual(@as(usize, 0), try compare(a, f, &pristine, 1104, 8, 4, 78));
}
