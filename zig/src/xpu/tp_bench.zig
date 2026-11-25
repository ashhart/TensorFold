//! Phase 1 benchmark & bit-exactness verification for Level Zero P2P TP=2 and TP=4 collectives on Arc Pro Xe2.
const std = @import("std");
const ze = @import("ze");
const device_id = @import("device_id");
const tp_mod = @import("tp.zig");
const kb = @import("xpu_tp_kernels");

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn f32ToF16Rte(v: f32) u16 {
    const f16_val: f16 = @floatCast(v);
    return @bitCast(f16_val);
}

fn runSuite(tp_deg: usize) !void {
    const alloc = std.heap.c_allocator;
    var gpus: [4]ze.Gpu = undefined;
    device_id.openPeers(gpus[0..tp_deg]) catch |err| {
        std.debug.print("skipped TP={d}: {s}\n", .{ tp_deg, @errorName(err) });
        return;
    };
    defer ze.Gpu.closePeers(gpus[0..tp_deg]);

    var lists: [4]ze.H = undefined;
    for (0..tp_deg) |r| lists[r] = try gpus[r].immediate(0, 0);

    const max_elems: usize = 4096 * 5120; // up to 4096 rows * 5120 fp16 = 40 MB
    const max_bytes: usize = max_elems * 2;
    var comm = try tp_mod.TpComm.init(gpus[0..tp_deg], &kb.tp, max_bytes);

    var dsts: [4]*anyopaque = undefined;
    for (0..tp_deg) |r| dsts[r] = try gpus[r].deviceAlloc(max_bytes);

    // 1. Strict rank-ordered bit-exactness check (with non-associative float magnitudes!)
    {
        const test_n: usize = 1024 * 5120; // > 512 KB so we also exercise 2-shot ReduceScatter + AllGather!
        var host_in: [4][]u16 = undefined;
        for (0..tp_deg) |r| {
            host_in[r] = try alloc.alloc(u16, test_n);
            var prng = std.Random.DefaultPrng.init(0x12345678 + @as(u64, r) * 99991);
            const rnd = prng.random();
            for (host_in[r], 0..) |*x, idx| {
                // Inject cancellation/non-associative patterns every 4th element
                const f: f16 = switch ((idx + r) % 4) {
                    0 => 1024.0,
                    1 => -1024.0,
                    2 => @floatCast((rnd.float(f32) - 0.5) * 2.0),
                    else => @floatCast((rnd.float(f32) - 0.5) * 0.01),
                };
                x.* = @bitCast(f);
            }
        }
        defer for (0..tp_deg) |r| alloc.free(host_in[r]);

        const ref = try alloc.alloc(u16, test_n);
        defer alloc.free(ref);
        for (0..test_n) |i| {
            var s: f32 = 0.0;
            for (0..tp_deg) |r| {
                const h: f16 = @bitCast(host_in[r][i]);
                s += @as(f32, h);
            }
            ref[i] = f32ToF16Rte(s);
        }

        // Check both small (1-shot, 16384 elems = 32 KB) and large (2-shot RS+AG, test_n = 10 MB)
        for ([_]usize{ 16384, test_n }) |cnt| {
            for (0..tp_deg) |r| {
                const sb = comm.currentSendBuf(r);
                try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], sb, host_in[r].ptr, cnt * 2, null, 0, null));
            }
            try comm.allReduceF16(gpus[0..tp_deg], lists[0..tp_deg], null, dsts[0..tp_deg], cnt);
            try comm.syncAll(gpus[0..tp_deg], lists[0..tp_deg]);

            const got = try alloc.alloc(u16, cnt);
            defer alloc.free(got);
            for (0..tp_deg) |r| {
                try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], got.ptr, dsts[r], cnt * 2, null, 0, null));
                try ze.check(gpus[r].api.zeCommandListHostSynchronize(lists[r], std.math.maxInt(u64)));
                for (0..cnt) |i| {
                    if (got[i] != ref[i]) {
                        std.debug.print("MISMATCH TP={d} rank={d} cnt={d} idx={d}: got=0x{x} ref=0x{x}\n", .{ tp_deg, r, cnt, i, got[i], ref[i] });
                        return error.BitMismatch;
                    }
                }
            }
        }
        std.debug.print("=== TP={d}: bit-exactness PASSED (1-shot 32 KB & 2-shot 10 MB, all {d} ranks == rank-ordered FP32 ref) ===\n", .{ tp_deg, tp_deg });
    }

    // 2. Latency & Bandwidth benchmarks (128 pipelined AllReduces per forward pass = 64 layers * 2)
    const cases = [_]struct { label: []const u8, elems: usize, steps: usize, reps: usize }{
        .{ .label = "decode 1 row   (10 KB)", .elems = 1 * 5120, .steps = 128, .reps = 20 },
        .{ .label = "verify 8 rows  (80 KB)", .elems = 8 * 5120, .steps = 128, .reps = 20 },
        .{ .label = "tree 16 rows  (160 KB)", .elems = 16 * 5120, .steps = 128, .reps = 20 },
        .{ .label = "prefill 512r  (5.0 MB)", .elems = 512 * 5120, .steps = 16, .reps = 10 },
        .{ .label = "prefill 1024r  (10 MB)", .elems = 1024 * 5120, .steps = 16, .reps = 10 },
        .{ .label = "prefill 4096r  (40 MB)", .elems = 4096 * 5120, .steps = 8, .reps = 5 },
    };

    for (cases) |tc| {
        // Warmup
        for (0..2) |_| {
            for (0..tc.steps) |_| {
                try comm.allReduceF16(gpus[0..tp_deg], lists[0..tp_deg], null, dsts[0..tp_deg], tc.elems);
            }
            try comm.syncAll(gpus[0..tp_deg], lists[0..tp_deg]);
        }
        var best_ns: u64 = std.math.maxInt(u64);
        var total_ns: u64 = 0;
        for (0..tc.reps) |_| {
            const t0 = nowNs();
            for (0..tc.steps) |_| {
                try comm.allReduceF16(gpus[0..tp_deg], lists[0..tp_deg], null, dsts[0..tp_deg], tc.elems);
            }
            try comm.syncAll(gpus[0..tp_deg], lists[0..tp_deg]);
            const dt = nowNs() - t0;
            if (dt < best_ns) best_ns = dt;
            total_ns += dt;
        }
        const mean_step_us = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(tc.reps * tc.steps)) / 1e3;
        const best_step_us = @as(f64, @floatFromInt(best_ns)) / @as(f64, @floatFromInt(tc.steps)) / 1e3;
        const bytes = @as(f64, @floatFromInt(tc.elems * 2));
        // Algbw = payload_bytes / time_s; Busbw = algbw * 2 * (tp - 1) / tp
        const alg_gbs = bytes / (best_step_us * 1e-6) / 1e9;
        const bus_gbs = alg_gbs * 2.0 * @as(f64, @floatFromInt(tp_deg - 1)) / @as(f64, @floatFromInt(tp_deg));
        const per_layer64_ms = best_step_us * 128.0 / 1e3;
        std.debug.print("  {s}: best={d:6.2} us/op (mean={d:6.2} us) | 128-op/tok={d:5.2} ms | algbw={d:5.2} GB/s | busbw={d:5.2} GB/s\n", .{
            tc.label, best_step_us, mean_step_us, per_layer64_ms, alg_gbs, bus_gbs,
        });
    }
}

pub fn main() !void {
    try runSuite(2);
    try runSuite(4);
}
