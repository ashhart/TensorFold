//! `tf-tp-test tcp-hip`: `tcp-host`'s checks and timings on HIP memory, staged through the TCP Kind (TF_TP_LINK).
const std = @import("std");
const tp = @import("tp");

/// The sizes `tcp-host` times, a rank's shard in bytes.
const sizes = [_]usize{ 16, 4096, 8192, 32768, 131072, 262144, 1048576, 4194304 };
const edges = [_]usize{ 1, 2, 3, 4, 7, 8, 15, 31, 63, 255, 1023, 3145, 65535, 65536, 65537 };
const max_bytes = 4 << 20;

pub fn run(cfg: tp.Config, gpa: std.mem.Allocator) !u8 {
    if (cfg.world != 2) {
        std.debug.print("tcp-hip runs two ranks (TF_TP_WORLD=2)\n", .{});
        return 2;
    }
    var rig = tp.hip_rig.Rig.open() catch {
        std.debug.print("FAIL tcp-hip rank {d}: no HIP runtime (libamdhip64.so) or no GPU\n", .{cfg.rank});
        return 1;
    };
    defer rig.close();
    try rig.use(cfg.device);
    var hip = try tp.stage_hip.Hip.init(&rig.lib);
    defer hip.deinit();
    var link_buf: [256]u8 = undefined;
    const link = cfg.linkLabel(&link_buf);
    var b = try tp.Bootstrap.open(cfg);
    defer b.close();
    const plan = b.releasePlan();
    var fds = [2]tp.sock.Fd{ -1, -1 };
    fds[1 - cfg.rank] = plan[if (cfg.rank == 0) 1 else 0];
    var net = tp.tcp.Rank.init(gpa, cfg.rank, 2, &fds, .{});
    defer net.deinit();
    var r: tp.stage.Rank = undefined;
    try r.open(&net, hip.device(), .{});
    defer r.deinit();
    const c = r.iface();
    const peer: u32 = 1 - cfg.rank;
    const s = try rig.stream();
    defer rig.destroy(s);
    const in = try rig.alloc(max_bytes);
    defer rig.free(in);
    const out = try rig.alloc(2 * max_bytes);
    defer rig.free(out);
    const host = try gpa.alloc(u8, 2 * max_bytes);
    defer gpa.free(host);
    const want = try gpa.alloc(u8, 3 * max_bytes); // the sum, then rank 0's and rank 1's inputs
    defer gpa.free(want);
    // 1. bit for bit against the reference: every timed size, the edges, and a spread up to exactly 4 MiB
    var lens: [400]usize = undefined;
    for (&lens, 0..) |*l, k| l.* = if (k < sizes.len) sizes[k] else if (k < sizes.len + edges.len) edges[k - sizes.len] else 1 + (k * 7919 * 13) % max_bytes;
    lens[lens.len - 1] = max_bytes;
    var gathers: u32 = 0;
    var sums: u32 = 0;
    var trades: u32 = 0;
    for (lens, 0..) |len, k| {
        tp.reference.fill(host[0..len], .u8, cfg.rank, k);
        try rig.put(in, host[0..len]);
        try c.allGather(in, out, len, .u8, s);
        try settle(&rig, c, s);
        try rig.get(host[0 .. 2 * len], out);
        for (0..2) |q| {
            tp.reference.fill(want[0..len], .u8, @intCast(q), k);
            if (!std.mem.eql(u8, host[q * len ..][0..len], want[0..len])) return fail(cfg.rank, "all-gather", k, len);
        }
        gathers += 1;
        inline for (.{ tp.DType.f32, tp.DType.bf16 }) |t| {
            const n = (len / t.size()) * t.size();
            if (n > 0) {
                tp.reference.fill(host[0..n], t, cfg.rank, k);
                try rig.put(in, host[0..n]);
                tp.reference.fill(want[max_bytes..][0..n], t, 0, k);
                tp.reference.fill(want[2 * max_bytes ..][0..n], t, 1, k);
                const views = [_][]const u8{ want[max_bytes..][0..n], want[2 * max_bytes ..][0..n] };
                tp.reference.reduce(want[0..n], &views, t, .sum);
                try c.allReduce(in, out, n / t.size(), t, .sum, s);
                try settle(&rig, c, s);
                try rig.get(host[0..n], out);
                if (!std.mem.eql(u8, host[0..n], want[0..n])) return fail(cfg.rank, @tagName(t) ++ " sum", k, n);
                sums += 1;
            }
        }
        // exchange, then send and recv, and a variable-length gather (whole strides here), on every fourth length
        if (k % 4 == 0) {
            tp.reference.fill(host[0..len], .u8, cfg.rank, k + 1);
            try rig.put(in, host[0..len]);
            try c.exchange(in, out, len, .u8, peer, s);
            if (cfg.rank == 0) try c.send(in, len, .u8, 1, s) else try c.recv(out + len, len, .u8, 0, s);
            try settle(&rig, c, s);
            try rig.get(host[0 .. 2 * len], out);
            tp.reference.fill(want[0..len], .u8, peer, k + 1);
            if (!std.mem.eql(u8, host[0..len], want[0..len])) return fail(cfg.rank, "exchange", k, len);
            tp.reference.fill(want[0..len], .u8, 0, k + 1);
            if (cfg.rank == 1 and !std.mem.eql(u8, host[len..][0..len], want[0..len])) return fail(cfg.rank, "send/recv", k, len);
            const vl = [2]i32{ @intCast(len), @intCast(len / 3) };
            tp.reference.fill(host[0..len], .u8, cfg.rank, k + 2);
            try rig.put(in, host[0..len]);
            try c.allGatherV(in, out, len, @intFromPtr(&vl), .u8, s);
            try settle(&rig, c, s);
            try rig.get(host[0 .. 2 * len], out);
            for (0..2) |q| {
                const ql: usize = @intCast(vl[q]);
                tp.reference.fill(want[0..ql], .u8, @intCast(q), k + 2);
                if (!std.mem.eql(u8, host[q * len ..][0..ql], want[0..ql])) return fail(cfg.rank, "variable all-gather", k, len);
            }
            trades += 1;
        }
    }
    try c.check();
    std.debug.print("PASS tcp-hip rank {d}/2: {d} all-gathers (1 B to 4 MiB), {d} fp32 and bf16 sums, {d} exchanges, send/recv and variable all-gathers bit-exact from HIP memory over {s}\n", .{ cfg.rank, gathers, sums, trades, link });
    // 2. by size, median and p90 of a call queued then its stream synchronised, and of the queueing alone
    var times: [500]u64 = undefined;
    var queued: [500]u64 = undefined;
    for (sizes) |len| {
        inline for (.{ "all_gather", "exchange" }) |what| {
            for (&times, &queued) |*t, *e| {
                const t0 = tp.sock.nowNs();
                if (comptime std.mem.eql(u8, what, "all_gather")) try c.allGather(in, out, len, .u8, s) else try c.exchange(in, out, len, .u8, peer, s);
                e.* = tp.sock.nowNs() - t0;
                try settle(&rig, c, s);
                t.* = tp.sock.nowNs() - t0;
            }
            std.mem.sort(u64, &times, {}, std.sort.asc(u64));
            std.mem.sort(u64, &queued, {}, std.sort.asc(u64));
            if (cfg.rank == 0) std.debug.print("RESULT {{\"backend\":\"tcp-hip\",\"link\":\"{s}\",\"world\":2,\"call\":\"{s}\",\"bytes\":{d},\"p50_us\":{d:.2},\"p90_us\":{d:.2},\"gb_s_p50\":{d:.2},\"queue_p50_us\":{d:.2}}}\n", .{
                link,                                                                        what,            len, us(times[250]), us(times[450]),
                @as(f64, @floatFromInt(len)) / @as(f64, @floatFromInt(@max(times[250], 1))), us(queued[250]),
            });
        }
    }
    try c.check();
    try b.agree("done", "tcp-hip end");
    return 0;
}

/// The stream's calls done, and the communicator still sound: a failure is its error by name, never compared bytes.
fn settle(rig: *tp.hip_rig.Rig, c: tp.Collective, s: tp.collective.Stream) !void {
    try rig.sync(s);
    try c.check();
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

fn fail(rank: u32, what: []const u8, k: usize, len: usize) u8 {
    std.debug.print("FAIL tcp-hip rank {d}: {s} {d} of {d} bytes differs from the reference\n", .{ rank, what, k, len });
    return 1;
}
