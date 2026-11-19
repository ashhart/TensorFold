//! Ling MLA kernels-alone: rope_q, rope_cache, gate_expand vs host.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const c = @import("ling_mla_gpu_common.zig");

fn checkRopeQ() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    const H = s.heads;
    var seed: u64 = 0x7271;
    const q = try c.fill(&seed, rows * H * mla.qd, a);
    defer a.free(q);
    c.roundSlice(q);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);
    const g = try a.alloc(f32, rows * H);
    defer a.free(g);
    @memset(g, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try c.Gpu.init();
    defer gpu.deinit();
    var l = try c.Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, g);
    defer l.deinit();
    l.fillRows(0, rows);
    l.fillCs(rows);
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.rope_q);
    e.setBuffer(l.b_q, 0, 0);
    e.setValue(aa, 1);
    e.setBuffer(l.b_cs, 0, 2);
    e.dispatchGroups(mtl.Size.of(H, rows, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("rope_q failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    const got = l.b_q.slice(u16, q.len);
    var worst: f32 = 0;
    var bits_miss: usize = 0;
    for (0..rows) |t| {
        for (0..H) |h| {
            var z64: [mla.rope_dim]f64 = undefined;
            var out: [mla.rope_dim]f64 = undefined;
            const base = (t * H + h) * mla.qd + mla.nope;
            for (0..mla.rope_dim) |i| z64[i] = q[base + i];
            mla.rotate(&out, &z64, @intCast(t), s.theta);
            for (0..mla.rope_dim) |i| {
                const want = c.toBf16(@floatCast(out[i]));
                const gbits = got[base + i];
                if (want != gbits) bits_miss += 1;
                worst = @max(worst, @abs(c.fromBf16(want) - c.fromBf16(gbits)));
            }
        }
    }
    std.debug.print("rope_q vs host rotate: bits miss {d} worst abs {e}\n", .{ bits_miss, worst });
    if (worst > 2e-3) return error.RopeQ;
}

fn checkRopeCache() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    var seed: u64 = 0x7263;
    const rope_in = try c.fill(&seed, rows * mla.rope_dim, a);
    defer a.free(rope_in);
    c.roundSlice(rope_in);
    const H = s.heads;
    const q = try a.alloc(f32, rows * H * mla.qd);
    defer a.free(q);
    @memset(q, 0);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);
    const g = try a.alloc(f32, rows * H);
    defer a.free(g);
    @memset(g, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try c.Gpu.init();
    defer gpu.deinit();
    var l = try c.Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, g);
    defer l.deinit();
    const cache = l.b_cache.slice(u16, rows * mla.key);
    @memset(cache, 0);
    for (0..rows) |t| {
        for (0..mla.rope_dim) |i| cache[t * mla.key + mla.lat + i] = c.toBf16(rope_in[t * mla.rope_dim + i]);
    }
    l.fillRows(0, rows);
    l.fillCs(rows);
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.rope_cache);
    e.setBuffer(l.b_rows, 0, 0);
    e.setValue(aa, 1);
    e.setBuffer(l.b_cs, 0, 2);
    e.useResource(l.b_cache, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("rope_cache failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    var worst: f32 = 0;
    var bits_miss: usize = 0;
    for (0..rows) |t| {
        var z64: [mla.rope_dim]f64 = undefined;
        var out: [mla.rope_dim]f64 = undefined;
        for (0..mla.rope_dim) |i| z64[i] = rope_in[t * mla.rope_dim + i];
        mla.rotate(&out, &z64, @intCast(t), s.theta);
        for (0..mla.rope_dim) |i| {
            const want = c.toBf16(@floatCast(out[i]));
            const gbits = cache[t * mla.key + mla.lat + i];
            if (want != gbits) bits_miss += 1;
            worst = @max(worst, @abs(c.fromBf16(want) - c.fromBf16(gbits)));
        }
    }
    std.debug.print("rope_cache vs host rotate: bits miss {d} worst abs {e}\n", .{ bits_miss, worst });
    if (worst > 2e-3) return error.RopeCache;
}

fn checkExpand() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    const H = s.heads;
    var seed: u64 = 0x6765;
    const gate = try c.fill(&seed, rows * H, a);
    defer a.free(gate);
    c.roundSlice(gate);
    const q = try a.alloc(f32, rows * H * mla.qd);
    defer a.free(q);
    @memset(q, 0);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try c.Gpu.init();
    defer gpu.deinit();
    var l = try c.Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, gate);
    defer l.deinit();
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.gate_expand);
    e.setBuffer(l.b_g, 0, 0);
    e.setBuffer(l.b_g2, 0, 1);
    e.setValue(aa, 2);
    e.dispatchThreads(mtl.Size.of(H * mla.vd, rows, 1), mtl.Size.of(128, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("gate_expand failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    const g_bits = l.b_g.slice(u16, rows * H);
    const want = try a.alloc(u16, rows * H * mla.vd);
    defer a.free(want);
    mla.expandGate(want, g_bits, rows, H);
    try c.expectBitsEqual("gate_expand", l.b_g2.slice(u16, want.len), want);
}

test "Ling rope_q GPT-J halves vs host rotate" {
    try checkRopeQ();
}

test "Ling rope_cache GPT-J halves vs host rotate" {
    try checkRopeCache();
}

test "Ling gate_expand (T,H) to (T,H*128) bits-equal" {
    try checkExpand();
}
