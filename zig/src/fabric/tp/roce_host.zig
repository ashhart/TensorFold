//! `tf-tp-test roce-host`: the RoCE wire and proxy between two hosts without a GPU, checked bit for bit, then timed.
const std = @import("std");
const tp = @import("tp");
const roce = tp.roce;
const rv = tp.roce_verbs;

const sizes = [_]usize{ 16, 4096, 8192, 16384, 65536, 262144 };

pub fn run(cfg: tp.Config, gpa: std.mem.Allocator) !u8 {
    var b = try tp.Bootstrap.open(cfg);
    defer b.close();
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var n: usize = 0;
    if (cfg.roce_hca) |text| n = try rv.parseSpecs(text, &specs) else {
        var v = try tp.Verbs.open(null);
        defer v.close();
        n = rv.autoSpecs(&v, &specs, cfg.roce_hcas);
    }
    const r = try roce.Roce.init(null, &b, "", .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = 256 * 1024, .timeout_ns = 10 * std.time.ns_per_s });
    defer r.deinit();
    const in = try gpa.alloc(u8, 256 * 1024);
    defer gpa.free(in);
    const peer = try gpa.alloc(u8, 256 * 1024);
    defer gpa.free(peer);
    const out = try gpa.alloc(u8, 512 * 1024);
    defer gpa.free(out);
    // 1. every size from 1 byte to the slot, both slots, bit for bit
    var checked: u32 = 0;
    for (0..1000) |k| {
        const len = 1 + (k * 7919) % (256 * 1024);
        tp.reference.fill(in[0..len], .u8, cfg.rank, k);
        tp.reference.fill(peer[0..len], .u8, 1 - cfg.rank, k);
        try r.hostGather(in[0..len], out[0 .. 2 * len]);
        const mine_at = cfg.rank * len;
        const peer_at = (1 - cfg.rank) * len;
        if (!std.mem.eql(u8, out[mine_at..][0..len], in[0..len]) or !std.mem.eql(u8, out[peer_at..][0..len], peer[0..len])) {
            std.debug.print("FAIL roce-host gather {d} of {d} bytes\n", .{ k, len });
            return 1;
        }
        checked += 1;
    }
    try r.check();
    std.debug.print("PASS roce-host: {d} gathers of 1 B to 256 KiB bit-exact over {s}\n", .{ checked, @tagName(cfg.roce_wire) });
    // 2. one exchange's host-to-host time by size: median and p90 of 2,000 back-to-back ops
    var times: [2000]u64 = undefined;
    for (sizes) |len| {
        for (&times) |*t| {
            const t0 = tp.sock.nowNs();
            try r.hostGather(in[0..len], out[0 .. 2 * len]);
            t.* = tp.sock.nowNs() - t0;
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        if (cfg.rank == 0) std.debug.print("RESULT {{\"backend\":\"roce-host\",\"wire\":\"{s}\",\"bytes\":{d},\"p50_us\":{d:.2},\"p90_us\":{d:.2}}}\n", .{ @tagName(cfg.roce_wire), len, @as(f64, @floatFromInt(times[1000])) / 1e3, @as(f64, @floatFromInt(times[1800])) / 1e3 });
    }
    try r.check();
    try b.agree("done", "roce-host end");
    return 0;
}
