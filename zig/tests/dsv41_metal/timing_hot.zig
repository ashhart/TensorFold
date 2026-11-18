//! TF_DSV41_TIMING_HOT: a REAL layer replayed back to back on a busy GPU, as in a round; what each wave or launch adds.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const recorder = @import("recorder.zig");
const timing = @import("timing_real.zig");

const counters = mtl.counters;
const objc = mtl.objc;

const src =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void hot_spin(device uint* o [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    \\  float a = float(gid & 7) * 1e-3f, b = 1.0f;
    \\  for (uint i = 0; i < iters; ++i) { a = fma(a, 0.999f, 1e-4f); b = fma(b, a, 0.5f); }
    \\  if (a == 12345.0f && b == 0.0f) o[gid & 1023] = 1;
    \\}
    \\// each group reads its 16 KB of w; the words are kept only in a sum no one reads
    \\kernel void hot_warm(const device uint4* w [[buffer(0)]], device uint* o [[buffer(1)]], uint tg [[threadgroup_position_in_grid]],
    \\                     uint t [[thread_index_in_threadgroup]]) {
    \\  const device uint4* b = w + ulong(tg) * 1024;
    \\  const uint4 v = b[t] ^ b[t + 256] ^ b[t + 512] ^ b[t + 768];
    \\  if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9E3779B9u) o[tg & 1023] = v.x;
    \\}
    \\// the routed slots' matrix `m` (0 w1, 2 w3, 4 w2 of the six table words): 16 KB a group, from the plan on the GPU
    \\kernel void hot_warm_xp(const device ulong* table [[buffer(0)]], const device uint* slots [[buffer(1)]],
    \\                        const device uint* nslots [[buffer(2)]], device uint* o [[buffer(3)]], constant uint2& mb [[buffer(4)]],
    \\                        uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  if (tg.y >= nslots[0]) return;
    \\  const ulong at = (table[6 * slots[3 * tg.y] + mb.x] & ~ulong(15)) + ulong(tg.x) * 16384;
    \\  const device uint4* b = (const device uint4*)at;
    \\  if (ulong(tg.x) * 16384 + 16384 > mb.y) return;
    \\  const uint4 v = b[t] ^ b[t + 256] ^ b[t + 512] ^ b[t + 768];
    \\  if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9E3779B9u) o[tg.x & 1023] = v.x;
    \\}
;

/// A weight range read in wave `wave` of each replay; `xp` >= 0: the routed slots' matrix (table word 0, 2 or 4).
const Warm = struct { wave: usize, buf: mtl.Buffer, off: usize, len: usize, xp: i32 = -1 };

const Kit = struct {
    r: *fixture.Rig,
    junk: mtl.Buffer,
    sink: mtl.Buffer,
    samples: counters.SampleBuffer,
    pass: counters.ComputePass,
    spin: mtl.Pipeline,
    warm: mtl.Pipeline,
    warm_xp: mtl.Pipeline,
    fence: objc.Id,
    reps: usize,
    spin_iters: u32,
};

/// The encoder state a replay has set so far, so an extra dispatch can put it back.
const State = struct {
    pipe: ?objc.Id = null,
    bufs: [31]?recorder.Op = @splat(null),
    bytes: [31]?recorder.Op = @splat(null),

    fn note(s: *State, op: recorder.Op) void {
        switch (op) {
            .pipe => |p| s.pipe = p,
            .buf => |b| s.bufs[b.at] = op,
            .bytes => |b| s.bytes[b.at] = op,
            else => {},
        }
    }

    fn restore(s: *const State, rec: *const recorder.Recording, e: mtl.ComputeEncoder) void {
        if (s.pipe) |p| objc.msg(void, e.id, "setComputePipelineState:", .{p});
        for (s.bufs) |o| if (o) |op| objc.msg(void, e.id, "setBuffer:offset:atIndex:", .{ op.buf.b, op.buf.off, op.buf.at });
        for (s.bytes) |o| if (o) |op| e.setBytes(rec.data.items[op.bytes.from..][0..op.bytes.len], op.bytes.at);
    }
};

fn warmOne(k: *const Kit, e: mtl.ComputeEncoder, w: Warm) void {
    if (w.xp >= 0) {
        const r = k.r;
        const l = &r.layers[0];
        for (l.backing()) |b| e.useResource(b, 1);
        e.setPipeline(k.warm_xp);
        e.setBuffer(l.table, 0, 0);
        e.setBuffer(r.s.plan[0].buf, r.s.plan[0].off, 1);
        e.setBuffer(r.s.plan[1].buf, r.s.plan[1].off, 2);
        e.setBuffer(k.sink, 0, 3);
        const bytes: u32 = @intCast(r.c.expert_width * r.c.hidden / 2);
        e.setValue([2]u32{ @intCast(w.xp), bytes }, 4);
        e.dispatchGroups(mtl.Size.of(bytes / 16384, r.c.topk * 16, 1), mtl.Size.of(256, 1, 1));
        return;
    }
    const chunks = w.len / 16384;
    if (chunks == 0) return;
    e.setPipeline(k.warm);
    e.setBuffer(w.buf, w.off, 0);
    e.setBuffer(k.sink, 0, 1);
    e.dispatchGroups(mtl.Size.of(chunks, 1, 1), mtl.Size.of(256, 1, 1));
}

/// The recording with wave `drop`'s dispatches left out and each warm read put in its wave, then its barrier.
fn replay(k: *const Kit, rec: *const recorder.Recording, e: mtl.ComputeEncoder, drop: ?usize, warms: []const Warm) void {
    var st: State = .{};
    var w: usize = 0;
    var d: usize = 0; // dispatch index; a drop of 1000 + i leaves out dispatch i alone
    var placed = false;
    for (rec.ops.items) |op| {
        switch (op) {
            .barrier => {
                if (!placed) {
                    var any = false;
                    for (warms) |x| if (x.wave == w) {
                        warmOne(k, e, x);
                        any = true;
                    };
                    if (any) st.restore(rec, e);
                }
                placed = false;
                w += 1;
                e.barrier();
                continue;
            },
            .groups => |g| {
                if (drop == null or (drop.? != w and drop.? != 1000 + d)) e.dispatchGroups(g[0], g[1]);
                d += 1;
            },
            .threads => |g| {
                if (drop == null or (drop.? != w and drop.? != 1000 + d)) e.dispatchThreads(g[0], g[1]);
                d += 1;
            },
            .pipe => |p| objc.msg(void, e.id, "setComputePipelineState:", .{p}),
            .buf => |b| objc.msg(void, e.id, "setBuffer:offset:atIndex:", .{ b.b, b.off, b.at }),
            .bytes => |b| e.setBytes(rec.data.items[b.from..][0..b.len], b.at),
            .tgmem => |t| e.setThreadgroupMemory(t.len, t.at),
            .use => |u| objc.msg(void, e.id, "useResource:usage:", .{ u.r, u.usage }),
        }
        st.note(op);
    }
    for (warms) |x| if (x.wave == w) warmOne(k, e, x);
}

/// One sampled run: the flush, a spin encoder, then `reps` replays in one concurrent encoder; seconds a replay.
fn once(k: *Kit, rec: *const recorder.Recording, drop: ?usize, warms: []const Warm) !f64 {
    const r = k.r;
    timing.flush(r, k.junk);
    const pool = objc.Pool.push();
    defer pool.pop();
    const before = counters.sampleClocks(r.device);
    const cb = r.queue.commandBuffer();
    if (k.spin_iters > 0) {
        const e0 = cb.compute(.serial);
        e0.setPipeline(k.spin);
        e0.setBuffer(k.sink, 0, 0);
        e0.setValue(k.spin_iters, 1);
        e0.dispatchThreads(mtl.Size.of(80 * 1024, 1, 1), mtl.Size.of(256, 1, 1));
        objc.msg(void, e0.id, "updateFence:", .{k.fence});
        e0.end();
    }
    const e = try k.pass.compute(cb, k.samples, 0, 1);
    if (k.spin_iters > 0) objc.msg(void, e.id, "waitForFence:", .{k.fence});
    for (0..k.reps) |_| {
        replay(k, rec, e, drop, warms);
        e.barrier();
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| std.debug.print("command buffer failed: {s}\n", .{msg});
    const res = try k.samples.resolveCompleted(2);
    defer res.deinit();
    const after = counters.sampleClocks(r.device);
    const cpu: f64 = @floatFromInt(after.cpu - before.cpu);
    const gpu: f64 = @floatFromInt(after.gpu - before.gpu);
    const v = res.values();
    return @as(f64, @floatFromInt(v[1].timestamp -% v[0].timestamp)) * cpu / gpu * 1e-9 / @as(f64, @floatFromInt(k.reps));
}

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// A weight by the forward's name, whole or a fraction of its bytes ("wq_b", "wo_b@0.5:1").
fn weightOf(l: *const pkg.weights.Layer, spec: []const u8) !struct { buf: mtl.Buffer, off: usize, len: usize } {
    var name = spec;
    var lo: f64 = 0;
    var hi: f64 = 1;
    if (std.mem.indexOfScalar(u8, spec, '@')) |at| {
        name = spec[0..at];
        const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.BadWarm;
        lo = try std.fmt.parseFloat(f64, spec[at + 1 .. colon]);
        hi = try std.fmt.parseFloat(f64, spec[colon + 1 ..]);
    }
    const W = struct { n: []const u8, w: ?pkg.weights.Fp8 };
    const all = [_]W{
        .{ .n = "wq_a", .w = l.wq_a },       .{ .n = "wkv", .w = l.wkv },      .{ .n = "wq_b", .w = l.wq_b },
        .{ .n = "ix_wq_b", .w = l.ix_wq_b }, .{ .n = "wo_a", .w = l.wo_a },    .{ .n = "wo_b", .w = l.wo_b },
        .{ .n = "w1", .w = l.shared[0] },    .{ .n = "w2", .w = l.shared[1] }, .{ .n = "w3", .w = l.shared[2] },
    };
    for (all) |x| if (std.mem.eql(u8, x.n, name)) {
        const w = x.w orelse return error.NoSuchWeight;
        const bytes = @as(usize, w.k) * w.n;
        const a: usize = @intFromFloat(@as(f64, @floatFromInt(bytes)) * lo);
        const b: usize = @intFromFloat(@as(f64, @floatFromInt(bytes)) * hi);
        const a16 = a / 16384 * 16384;
        return .{ .buf = w.w.buf, .off = w.w.off + a16, .len = b - a16 };
    };
    return error.NoSuchWeight;
}

/// TF_DSV41_HOT_WARMS: variants separated by ';', each a list of "wave:weight" reads, e.g. "0:wq_b,5:wo_a;10:w1".
fn parseWarms(a: std.mem.Allocator, l: *const pkg.weights.Layer, text: []const u8) ![]const []const Warm {
    var out: std.ArrayList([]const Warm) = .empty;
    var vit = std.mem.tokenizeScalar(u8, text, ';');
    while (vit.next()) |v| {
        var list: std.ArrayList(Warm) = .empty;
        var it = std.mem.tokenizeScalar(u8, v, ',');
        while (it.next()) |item| {
            const colon = std.mem.indexOfScalar(u8, item, ':') orelse return error.BadWarm;
            const wave = try std.fmt.parseInt(usize, item[0..colon], 10);
            const name = item[colon + 1 ..];
            const xps = [_][]const u8{ "xp_w1", "xp_w3", "xp_w2" };
            var xp: i32 = -1;
            for (xps, 0..) |x, i| if (std.mem.eql(u8, x, name)) {
                xp = @intCast(2 * i);
            };
            if (xp >= 0) {
                try list.append(a, .{ .wave = wave, .buf = l.table, .off = 0, .len = 0, .xp = xp });
                continue;
            }
            const w = try weightOf(l, name);
            try list.append(a, .{ .wave = wave, .buf = w.buf, .off = w.off, .len = w.len });
        }
        try out.append(a, list.items);
    }
    return out.items;
}

pub fn run(a: std.mem.Allocator, r: *fixture.Rig, names: *const timing.Names, p0: u32, rows: u32, n: usize) !void {
    const lib = try mtl.Library.fromSource(r.device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var k: Kit = .{
        .r = r,
        .junk = try r.device.buffer(timing.flush_bytes, mtl.ResourceOptions.private),
        .sink = try r.device.buffer(4096, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked),
        .samples = try counters.SampleBuffer.init(r.device, 4),
        .pass = try counters.ComputePass.init(),
        .spin = try mtl.Pipeline.init(r.device, lib, "hot_spin", false),
        .warm = try mtl.Pipeline.init(r.device, lib, "hot_warm", false),
        .warm_xp = try mtl.Pipeline.init(r.device, lib, "hot_warm_xp", false),
        .fence = objc.msg(objc.Id, r.device.id, "newFence", .{}),
        .reps = if (env("TF_DSV41_HOT_REPS")) |v| try std.fmt.parseInt(usize, v, 10) else 8,
        .spin_iters = if (env("TF_DSV41_HOT_SPIN")) |v| try std.fmt.parseInt(u32, v, 10) else 100000,
    };
    defer {
        k.junk.deinit();
        k.sink.deinit();
        k.samples.deinit();
        k.pass.deinit();
        k.spin.deinit();
        k.warm.deinit();
        k.warm_xp.deinit();
        objc.release(k.fence);
    }
    objc.msg(void, k.pass.id, "setDispatchType:", .{@as(usize, @backingInt(mtl.DispatchType.concurrent))});
    var span = [1]pkg.scratch.Span{.{ .p0 = p0, .n = rows }};
    try r.s.writeSpans(&r.c, &span);
    const ctx = r.ctx(&r.slot, p0, rows);
    var rec: recorder.Recording = .{ .a = a };
    defer rec.deinit();
    const s = r.streams();
    const pr = r.pres();
    // in and out on the same streams: every replay reads what the one before wrote (only the time counts)
    const re = recorder.start(&rec);
    try pkg.layer.encode(&ctx, re, &r.layers[0], .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
    recorder.stop(re);
    var nw: usize = 1;
    for (rec.ops.items) |op| nw += @intFromBool(op == .barrier);
    const warms = try parseWarms(a, &r.layers[0], env("TF_DSV41_HOT_WARMS") orelse "");
    const drops = env("TF_DSV41_HOT_DROPS") != null;
    const dropd = env("TF_DSV41_HOT_DROPD") != null;
    const nd = rec.dispatches();
    // variants: base, each warm list, each wave dropped, each dispatch dropped; interleaved over n runs
    const nv = 1 + warms.len + (if (drops) nw else 0) + (if (dropd) nd else 0);
    const w0 = 1 + warms.len + (if (drops) nw else 0);
    const t = try a.alloc(f64, nv * n);
    _ = try once(&k, &rec, null, &.{});
    for (0..n) |it| for (0..nv) |i| {
        t[i * n + it] = if (i == 0) try once(&k, &rec, null, &.{}) else if (i <= warms.len) try once(&k, &rec, null, warms[i - 1]) else if (i < w0) try once(&k, &rec, i - 1 - warms.len, &.{}) else try once(&k, &rec, 1000 + i - w0, &.{});
    };
    const us = 1e6;
    const base = timing.least(t[0..n]);
    std.debug.print("hot: layer {d}, {d} row(s), {d} replays a run after a spin of {d}, fastest (median) of {d}, us a layer:\n", .{ r.layers[0].layer, rows, k.reps, k.spin_iters, n });
    std.debug.print("  layer {d:.1} ({d:.1})\n", .{ base * us, timing.median(t[0..n]) * us });
    var vit = std.mem.tokenizeScalar(u8, env("TF_DSV41_HOT_WARMS") orelse "", ';');
    for (0..warms.len) |i| {
        const v = t[(1 + i) * n ..][0..n];
        const f = timing.least(v);
        std.debug.print("  warm {s}: {d:.1} ({d:.1}), {d:.1} less\n", .{ vit.next().?, f * us, timing.median(v) * us, (base - f) * us });
    }
    if (drops) {
        // each wave's dispatch names, for the drop lines
        var w: usize = 0;
        var pipe: ?objc.Id = null;
        var wn: std.ArrayList(std.ArrayList(u8)) = .empty;
        for (0..nw) |_| try wn.append(a, .empty);
        for (rec.ops.items) |op| switch (op) {
            .pipe => |p| pipe = p,
            .barrier => w += 1,
            .groups, .threads => {
                if (wn.items[w].items.len > 0) try wn.items[w].appendSlice(a, " | ");
                try wn.items[w].appendSlice(a, names.of(pipe));
            },
            else => {},
        };
        for (0..nw) |i| {
            const v = t[(1 + warms.len + i) * n ..][0..n];
            const f = timing.least(v);
            std.debug.print("  wave {d:>2} adds {d:>6.1} us (without it {d:.1}): {s}\n", .{ i, (base - f) * us, f * us, wn.items[i].items });
        }
    }
    if (dropd) {
        var i: usize = 0;
        var pipe: ?objc.Id = null;
        for (rec.ops.items) |op| switch (op) {
            .pipe => |p| pipe = p,
            .groups, .threads => {
                const v = t[(w0 + i) * n ..][0..n];
                const f = timing.least(v);
                std.debug.print("  dispatch {d:>2} adds {d:>6.1} us: {s}\n", .{ i, (base - f) * us, names.of(pipe) });
                i += 1;
            },
            else => {},
        };
    }
}
