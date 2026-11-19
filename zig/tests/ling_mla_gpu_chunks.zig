//! Ling MLA whole vs chunks, resume vs fresh, large-T and dump gates.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const c = @import("ling_mla_gpu_common.zig");
const dump = @import("ling_mla_gpu_dump.zig");

const k3_gpu_rows: u32 = 256;

fn checkChunks() !void {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try c.Inputs.make(a, s, rows, 0x6d6c61);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try c.Gpu.init();
    defer g.deinit();
    var whole = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer whole.deinit();
    try whole.dispatchMixer(g, 0, rows);
    var chunks = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer chunks.deinit();
    try chunks.dispatchMixer(g, 0, 2);
    try chunks.dispatchMixer(g, 2, 2);
    try c.expectBitsEqual("whole vs chunks", whole.y(), chunks.y());
}

fn checkResume() !void {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try c.Inputs.make(a, s, rows, 0x72657375);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try c.Gpu.init();
    defer g.deinit();
    var fresh = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer fresh.deinit();
    try fresh.dispatchMixer(g, 0, rows);
    var continued = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer continued.deinit();
    try continued.dispatchMixer(g, 0, 2);
    try continued.dispatchMixer(g, 2, 2);
    const W = s.outWidth();
    try c.expectBitsEqual("resume vs fresh (all rows)", fresh.y(), continued.y());
    try c.expectBitsEqual("resume vs fresh (last 2)", fresh.y()[2 * W ..], continued.y()[2 * W ..]);
}

fn checkRopeLargePos() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows_n: u32 = 2;
    const first_pos: u32 = 29998;
    const slot_keys: u32 = 30000;
    const H = s.heads;
    var seed: u64 = 0x72333030;
    const q = try c.fill(&seed, rows_n * H * mla.qd, a);
    defer a.free(q);
    c.roundSlice(q);
    const rope_in = try c.fill(&seed, rows_n * mla.rope_dim, a);
    defer a.free(rope_in);
    c.roundSlice(rope_in);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try c.Gpu.init();
    defer gpu.deinit();
    const shared = mtl.ResourceOptions.shared;
    const b_q = try gpu.device.buffer(q.len * 2, shared);
    defer b_q.deinit();
    c.putBf16(b_q, q);
    const b_rows = try gpu.device.buffer(rows_n * @sizeOf(c.MlaRow), shared);
    defer b_rows.deinit();
    const rows = b_rows.slice(c.MlaRow, rows_n);
    for (rows, 0..) |*r, i| r.* = .{ .slot = 0, .pos = first_pos + @as(u32, @intCast(i)) };
    const b_cs = try gpu.device.buffer(rows_n * mla.nfreq * 2 * 4, shared);
    defer b_cs.deinit();
    c.writeCs(b_cs.slice(f32, rows_n * mla.nfreq * 2), rows, s.theta);
    const b_cache = try gpu.device.buffer(@as(usize, slot_keys) * mla.key * 2, shared);
    defer b_cache.deinit();
    const cache = b_cache.slice(u16, slot_keys * mla.key);
    for (0..rows_n) |t| {
        const base = (first_pos + t) * mla.key + mla.lat;
        for (0..mla.rope_dim) |i| cache[base + i] = c.toBf16(rope_in[t * mla.rope_dim + i]);
    }
    const aa = c.MlaArgs{
        .cache = b_cache.gpuAddress(),
        .slot_keys = slot_keys,
        .heads = H,
        .head0 = 0,
        .rows = rows_n,
        .q_stride = H * mla.qd,
        .kv_stride = mla.key,
        .gate_stride = H * mla.vd,
        .out_stride = H * mla.vd,
        .eps = s.eps,
        .scale = @floatCast(s.scale()),
    };

    {
        const cb = gpu.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(gpu.rope_q);
        e.setBuffer(b_q, 0, 0);
        e.setValue(aa, 1);
        e.setBuffer(b_cs, 0, 2);
        e.dispatchGroups(mtl.Size.of(H, rows_n, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("rope_q large pos failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }
    const got_q = b_q.slice(u16, q.len);
    const nq = rows_n * H * mla.rope_dim;
    const metal_q = try a.alloc(f64, nq);
    defer a.free(metal_q);
    const f64_q = try a.alloc(f64, nq);
    defer a.free(f64_q);
    const bf_q = try a.alloc(f64, nq);
    defer a.free(bf_q);
    var k: usize = 0;
    for (0..rows_n) |t| {
        for (0..H) |h| {
            var z64: [mla.rope_dim]f64 = undefined;
            var out: [mla.rope_dim]f64 = undefined;
            const base = (t * H + h) * mla.qd + mla.nope;
            for (0..mla.rope_dim) |i| z64[i] = q[base + i];
            mla.rotate(&out, &z64, first_pos + @as(u32, @intCast(t)), s.theta);
            for (0..mla.rope_dim) |i| {
                f64_q[k] = out[i];
                bf_q[k] = c.fromBf16(c.toBf16(@floatCast(out[i])));
                metal_q[k] = c.fromBf16(got_q[base + i]);
                k += 1;
            }
        }
    }
    const q_gate = c.gateNamedF64("rope_q pos 29998-29999", metal_q, f64_q, bf_q);

    {
        const cb = gpu.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(gpu.rope_cache);
        e.setBuffer(b_rows, 0, 0);
        e.setValue(aa, 1);
        e.setBuffer(b_cs, 0, 2);
        e.useResource(b_cache, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("rope_cache large pos failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }
    const nc = rows_n * mla.rope_dim;
    const metal_c = try a.alloc(f64, nc);
    defer a.free(metal_c);
    const f64_c = try a.alloc(f64, nc);
    defer a.free(f64_c);
    const bf_c = try a.alloc(f64, nc);
    defer a.free(bf_c);
    k = 0;
    for (0..rows_n) |t| {
        var z64: [mla.rope_dim]f64 = undefined;
        var out: [mla.rope_dim]f64 = undefined;
        for (0..mla.rope_dim) |i| z64[i] = rope_in[t * mla.rope_dim + i];
        mla.rotate(&out, &z64, first_pos + @as(u32, @intCast(t)), s.theta);
        const base = (first_pos + t) * mla.key + mla.lat;
        for (0..mla.rope_dim) |i| {
            f64_c[k] = out[i];
            bf_c[k] = c.fromBf16(c.toBf16(@floatCast(out[i])));
            metal_c[k] = c.fromBf16(cache[base + i]);
            k += 1;
        }
    }
    const cache_gate = c.gateNamedF64("rope_cache pos 29998-29999", metal_c, f64_c, bf_c);
    q_gate catch |e| return e;
    cache_gate catch |e| return e;
}

fn checkLarge(rows: u32, left: u32, seed: u64, tag: []const u8) !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try c.Inputs.make(a, s, rows, seed);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try c.Gpu.init();
    defer g.deinit();
    var whole = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer whole.deinit();
    try whole.dispatchMixer(g, 0, rows);
    var parts = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer parts.deinit();
    try parts.dispatchMixer(g, 0, left);
    try parts.dispatchMixer(g, left, rows - left);
    var buf: [96]u8 = undefined;
    const t1 = try std.fmt.bufPrint(&buf, "{s} whole vs chunks", .{tag});
    try c.expectBitsEqual(t1, whole.y(), parts.y());
    var buf2: [96]u8 = undefined;
    const t2 = try std.fmt.bufPrint(&buf2, "{s} resume vs fresh", .{tag});
    try c.expectBitsEqual(t2, whole.y(), parts.y());
}

test "MLA mixer 4 rows equals 2+2 bits-equal" {
    try checkChunks();
}

test "MLA mixer resume vs fresh bits-equal" {
    try checkResume();
}

test "TFLINGA1 tiny layer 3 T=4 pos0=0 Metal and mixF64 vs python f64" {
    try dump.checkDumpFile(dump.dump_a1_t4_path, 0, "dump T=4 pos0=0", true);
}

test "TFLINGA1 tiny layer 3 T=2 pos0=30000 mixF64 vs python host-only" {
    try dump.checkDumpFile(dump.dump_a1_t2_path, 1, "dump T=2 pos0=30000", false);
}

test "Ling rope_q and rope_cache at pos 29998-29999 vs f64" {
    try checkRopeLargePos();
}

test "MLA mixer T=100 as 37+63 bits-equal" {
    try checkLarge(100, 37, 0x74313030, "T=100");
}

test "MLA mixer T=257 as 256+1 bits-equal (k3 gpu_rows)" {
    try checkLarge(257, k3_gpu_rows, 0x74323537, "T=257");
}
