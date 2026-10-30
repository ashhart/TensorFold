//! `tf-tp-test roce-host`: exact gathers and timing on the two-rank or versioned multirank wire.
const std = @import("std");
const tp = @import("tp");
const rv = tp.roce_verbs;

/// Shards a rank, bytes: the slot's edges and the engine's all-gathers (one 8 KiB bf16 row of 4,096 up to 128 rows).
const sizes = [_]usize{ 16, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };
const max_bytes = 1024 * 1024;

pub fn run(cfg: tp.Config, gpa: std.mem.Allocator) !u8 {
    if (cfg.world == 2) return runWith(tp.roce, cfg, gpa);
    return runWith(tp.roce_multi, cfg, gpa);
}

fn runWith(comptime roce: type, cfg: tp.Config, gpa: std.mem.Allocator) !u8 {
    var b = try tp.Bootstrap.open(cfg);
    defer b.close();
    const world: usize = cfg.world;
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var n: usize = 0;
    if (cfg.roce_wire == .verbs) n = try tp.roce_pick.resolve(cfg, &b, &specs);
    const r = try roce.Roce.init(null, &b, "", .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = max_bytes, .timeout_ns = 10 * std.time.ns_per_s });
    defer r.deinit();
    const in = try gpa.alloc(u8, max_bytes);
    defer gpa.free(in);
    const want = try gpa.alloc(u8, max_bytes);
    defer gpa.free(want);
    const out = try gpa.alloc(u8, world * max_bytes);
    defer gpa.free(out);
    var lens: [tp.bootstrap.max_ranks]u32 = undefined;
    // 1. every size from 1 byte to the slot, both slots, bit for bit; every third op with variable lengths
    var checked: u32 = 0;
    for (0..1500) |k| {
        const len = 1 + (k * 7919) % max_bytes;
        const v = k % 3 == 2;
        for (0..world) |q| lens[q] = @intCast(if (k % 5 == 0 and q == 0) 0 else (k * 31 + q * 17) % (len + 1));
        tp.reference.fill(in[0..len], .u8, cfg.rank, k);
        if (comptime roce == tp.roce) {
            try r.hostGatherV(in[0..len], out[0 .. world * len], if (v) .{ lens[0], lens[1] } else null);
        } else try r.hostGatherV(in[0..len], out[0 .. world * len], if (v) lens[0..world] else null);
        for (0..world) |q| {
            const l = if (v) lens[q] else len;
            tp.reference.fill(want[0..len], .u8, @intCast(q), k);
            if (!std.mem.eql(u8, out[q * len ..][0..l], want[0..l])) {
                std.debug.print("FAIL roce-host rank {d}: gather {d} of {d} bytes{s}: rank {d}'s shard differs\n", .{ cfg.rank, k, len, if (v) " (variable lengths)" else "", q });
                return 1;
            }
        }
        checked += 1;
    }
    try r.check();
    std.debug.print("PASS roce-host rank {d}/{d}: {d} gathers of 1 B to 1 MiB bit-exact over {s} ({d} HCA(s))\n", .{ cfg.rank, world, checked, @tagName(cfg.roce_wire), n });
    // 2. one exchange's host-to-host time by size: median and p90 of 2,000 back-to-back ops
    var times: [2000]u64 = undefined;
    for (sizes) |len| {
        for (&times) |*t| {
            const t0 = tp.sock.nowNs();
            try r.hostGather(in[0..len], out[0 .. world * len]);
            t.* = tp.sock.nowNs() - t0;
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        if (cfg.rank == 0) std.debug.print("RESULT {{\"backend\":\"roce-host\",\"wire\":\"{s}\",\"world\":{d},\"bytes\":{d},\"p50_us\":{d:.2},\"p90_us\":{d:.2}}}\n", .{ @tagName(cfg.roce_wire), world, len, @as(f64, @floatFromInt(times[1000])) / 1e3, @as(f64, @floatFromInt(times[1800])) / 1e3 });
    }
    try r.check();
    try b.agree("done", "roce-host end");
    return 0;
}
