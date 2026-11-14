//! DeepSeek-V4.1 on a Metal 4 queue against a Metal 3 queue, bit for bit: decode rounds, prompt chunks, timestamps.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const pt = @import("prompt_test.zig");

const testing = std.testing;
const fwd = pkg.forward;
const limit = 1280;
const look = 3;

const Out = struct {
    logits: []f32,
    taps: []u16,

    fn deinit(o: Out, a: std.mem.Allocator) void {
        a.free(o.logits);
        a.free(o.taps);
    }
};

fn reset(q: mtl.Queue, to: *const pkg.kv.Slot, from: *const pkg.kv.Slot) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = q.commandBuffer();
    to.copyFrom(cb, from);
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CommandBufferFailed;
}

/// `steps` decode rounds of `rows` rows from p0 through forward.run on `q`, the slot first copied from `pristine`.
fn rounds(a: std.mem.Allocator, f: *ff.Fwd, q: mtl.Queue, slot: *const pkg.kv.Slot, pristine: *const pkg.kv.Slot, hist: []const u32, p0: u32, rows: usize, steps: usize) !Out {
    const r = f.r;
    const c = &r.c;
    try reset(q, slot, pristine);
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    const out: Out = .{ .logits = try a.alloc(f32, rows * steps * V), .taps = try a.alloc(u16, rows * steps * TH) };
    for (0..steps) |k| {
        const at = k * rows;
        try fwd.prepare(&f.b, c, hist[look + at ..][0..rows], ff.lookbackAt(hist, look + at, look), f.lookup());
        const p: u32 = p0 + @as(u32, @intCast(at));
        try r.s.writeRows(c, p, @intCast(rows));
        const x = r.ctx(slot, p, @intCast(rows));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try fwd.run(&x, q, f.model(), &f.b);
        @memcpy(out.logits[at * V ..][0 .. rows * V], f.b.logitsOf(c, rows));
        @memcpy(out.taps[at * TH ..][0 .. rows * TH], slot.taps.?.slice(u16, rows * TH));
    }
    return out;
}

/// The same rounds on both queues: logits, taps and slot bytes that differ.
fn compare(a: std.mem.Allocator, f: *ff.Fwd, m3: mtl.Queue, m4: mtl.Queue, pristine: *const pkg.kv.Slot, rows: usize, steps: usize, seed: u64) !usize {
    const r = f.r;
    var s3 = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer s3.deinit();
    var s4 = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer s4.deinit();
    const hist = try ff.ids(a, &r.c, look + rows * steps, seed);
    defer a.free(hist);
    const o3 = try rounds(a, f, m3, &s3, pristine, hist, 1101, rows, steps);
    defer o3.deinit(a);
    const o4 = try rounds(a, f, m4, &s4, pristine, hist, 1101, rows, steps);
    defer o4.deinit(a);
    const lb = pt.bitsDiff(o3.logits, o4.logits);
    const tb = pt.halfDiff(o3.taps, o4.taps);
    const sb = pt.stateDiff(&s3, &s4);
    const word = if (lb + tb + sb == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal 4 == metal 3: {d} rounds of {d} rows: {d} logits, {d} tap values, {d} state bytes differ\n", .{ word, steps, rows, lb, tb, sb });
    return lb + tb + sb;
}

test "dsv41 metal: decode rounds on a Metal 4 queue equal a Metal 3 queue's bit for bit, 1 to 16 rows" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 71);
    defer f.deinit();
    if (!f.r.device.hasMetal4()) return error.SkipZigTest;
    const m3 = try f.r.device.queue();
    defer m3.deinit();
    const m4 = try f.r.device.queue4();
    defer m4.deinit();
    try testing.expect(m4.isMetal4() and !m3.isMetal4());
    var pristine = try pkg.kv.Slot.init(a, f.r.device, &f.r.c, limit);
    defer pristine.deinit();
    fixture.fillPast(&f.r.c, &pristine, 1101, 12);
    var bad: usize = 0;
    for ([_]usize{ 1, 2, 4, 6, 16 }) |rows| bad += try compare(a, f, m3, m4, &pristine, rows, 3, 300 + rows);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal: a prompt's chunks on a Metal 4 queue equal a Metal 3 queue's: logits, taps, slot and tail" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 72);
    defer f.deinit();
    if (!f.r.device.hasMetal4()) return error.SkipZigTest;
    const c = &f.r.c;
    const hist = try ff.ids(a, c, 900, 73);
    defer a.free(hist);
    var runs: [2]pt.Run = undefined;
    const qs = [2]mtl.Queue{ try f.r.device.queue(), try f.r.device.queue4() };
    defer for (qs) |q| q.deinit();
    for (&runs, qs) |*run, q| {
        run.* = try pt.Run.init(a, f, pt.head_opts, 0);
        var e = pt.env(f);
        e.queue = q;
        try run.p.run(e, &run.slot, &run.tail, hist);
    }
    defer for (&runs) |*run| run.deinit();
    const d = pt.runDiff(c, &runs[0], &runs[1]);
    const td = pt.tailDiff(&runs[0].tail, &runs[1].tail);
    const word = if (d[0] + d[1] + d[2] + td == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal 4 prompt == metal 3: {d} logits, {d} taps, {d} state bytes, {d} tail bytes differ\n", .{ word, d[0], d[1], d[2], td });
    try testing.expectEqual(@as(usize, 0), d[0] + d[1] + d[2] + td);
}

test "dsv41 metal: timestamped decode rounds on Metal 4 change no bit, and their stages are summed" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 74);
    defer f.deinit();
    if (!f.r.device.hasMetal4()) return error.SkipZigTest;
    const m3 = try f.r.device.queue();
    defer m3.deinit();
    const st = mtl.metal4.State.open(f.r.device.id, 2) orelse return error.SkipZigTest;
    const m4: mtl.Queue = .{ .id = st.queue, .m4 = st };
    defer m4.deinit();
    var pristine = try pkg.kv.Slot.init(a, f.r.device, &f.r.c, limit);
    defer pristine.deinit();
    fixture.fillPast(&f.r.c, &pristine, 1101, 13);
    // 8 warm rounds, then 2 stamped
    try testing.expectEqual(@as(usize, 0), try compare(a, f, m3, m4, &pristine, 1, 11, 400));
    const t = st.timeline orelse return error.NoTimeline;
    const r = t.rounds.items[0];
    std.debug.print("dsv41 metal 4 timeline: {s}: {d} stamped rounds, {d} encoders, {d} kernels, {d:.3} ms in stages a round\n", .{ r.name, r.done, r.encoders, r.kernels.count(), r.inside * 1e3 / @as(f64, @floatFromInt(r.done)) });
    try testing.expectEqual(@as(u32, 2), r.done);
    try testing.expect(r.encoders >= 1 and r.kernels.count() > 4 and r.inside > 0 and r.fits);
}
