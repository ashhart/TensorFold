//! `tf-tp-test tcp-host`: exact collectives and timing over the TCP Kind, each result naming its link (TF_TP_LINK).
const std = @import("std");
const tp = @import("tp");

/// Shards a rank, bytes: one bf16 row of 4,096 up to the engine's largest decode gathers and a 4 MiB prompt piece.
const sizes = [_]usize{ 16, 4096, 8192, 32768, 131072, 262144, 1048576, 4194304 };
/// Edges the bit-for-bit pass checks besides `sizes` and its own spread of lengths.
const edges = [_]usize{ 1, 2, 3, 4, 7, 8, 15, 31, 63, 255, 1023, 3145, 65535, 65536, 65537 };
const max_bytes = 4 << 20;

pub fn run(cfg: tp.Config, gpa: std.mem.Allocator) !u8 {
    if (cfg.world != 2) {
        std.debug.print("tcp-host runs two ranks (TF_TP_WORLD=2)\n", .{});
        return 2;
    }
    var link_buf: [256]u8 = undefined;
    const link = jsonSafe(cfg.link orelse "unnamed link (set TF_TP_LINK)", &link_buf);
    var b = try tp.Bootstrap.open(cfg);
    defer b.close();
    // the bootstrap's plan link: a socket to the peer already checked for version, world, rank and role
    const plan = b.releasePlan();
    var fds = [2]tp.sock.Fd{ -1, -1 };
    fds[1 - cfg.rank] = plan[if (cfg.rank == 0) 1 else 0];
    var r = tp.tcp.Rank.init(gpa, cfg.rank, 2, &fds, .{});
    defer r.deinit();
    const c = r.iface();
    const peer: u32 = 1 - cfg.rank;
    const in = try gpa.alloc(u8, max_bytes);
    defer gpa.free(in);
    const want = try gpa.alloc(u8, 3 * max_bytes); // the sum, then rank 0's and rank 1's inputs
    defer gpa.free(want);
    const out = try gpa.alloc(u8, 2 * max_bytes);
    defer gpa.free(out);
    // 1. bit for bit against the reference: every timed size, the edges, and a spread up to exactly 4 MiB
    var lens: [400]usize = undefined;
    for (&lens, 0..) |*l, k| l.* = if (k < sizes.len) sizes[k] else if (k < sizes.len + edges.len) edges[k - sizes.len] else 1 + (k * 7919 * 13) % max_bytes;
    lens[lens.len - 1] = max_bytes;
    var gathers: u32 = 0;
    var sums: u32 = 0;
    var trades: u32 = 0;
    for (lens, 0..) |len, k| {
        tp.reference.fill(in[0..len], .u8, cfg.rank, k);
        try c.allGather(@intFromPtr(in.ptr), @intFromPtr(out.ptr), len, .u8, null);
        for (0..2) |q| {
            tp.reference.fill(want[0..len], .u8, @intCast(q), k);
            if (!std.mem.eql(u8, out[q * len ..][0..len], want[0..len])) return fail(cfg.rank, "all-gather", k, len);
        }
        gathers += 1;
        inline for (.{ tp.DType.f32, tp.DType.bf16 }) |t| {
            const n = (len / t.size()) * t.size();
            if (n > 0) {
                tp.reference.fill(in[0..n], t, cfg.rank, k);
                tp.reference.fill(want[max_bytes..][0..n], t, 0, k);
                tp.reference.fill(want[2 * max_bytes ..][0..n], t, 1, k);
                const views = [_][]const u8{ want[max_bytes..][0..n], want[2 * max_bytes ..][0..n] };
                tp.reference.reduce(want[0..n], &views, t, .sum);
                try c.allReduce(@intFromPtr(in.ptr), @intFromPtr(out.ptr), n / t.size(), t, .sum, null);
                if (!std.mem.eql(u8, out[0..n], want[0..n])) return fail(cfg.rank, @tagName(t) ++ " sum", k, n);
                sums += 1;
            }
        }
        // exchange, then send and recv, and a variable-length gather with uneven lengths, on every fourth length
        if (k % 4 == 0) {
            tp.reference.fill(in[0..len], .u8, cfg.rank, k + 1);
            try c.exchange(@intFromPtr(in.ptr), @intFromPtr(out.ptr), len, .u8, peer, null);
            tp.reference.fill(want[0..len], .u8, peer, k + 1);
            if (!std.mem.eql(u8, out[0..len], want[0..len])) return fail(cfg.rank, "exchange", k, len);
            if (cfg.rank == 0) try c.send(@intFromPtr(in.ptr), len, .u8, 1, null) else try c.recv(@intFromPtr(out.ptr), len, .u8, 0, null);
            if (cfg.rank == 1) {
                tp.reference.fill(want[0..len], .u8, 0, k + 1);
                if (!std.mem.eql(u8, out[0..len], want[0..len])) return fail(cfg.rank, "send/recv", k, len);
            }
            const vl = [2]i32{ @intCast(len), @intCast(len / 3) };
            const mine_len: usize = @intCast(vl[cfg.rank]);
            tp.reference.fill(in[0..mine_len], .u8, cfg.rank, k + 2);
            try c.allGatherV(@intFromPtr(in.ptr), @intFromPtr(out.ptr), len, @intFromPtr(&vl), .u8, null);
            for (0..2) |q| {
                const ql: usize = @intCast(vl[q]);
                tp.reference.fill(want[0..ql], .u8, @intCast(q), k + 2);
                if (!std.mem.eql(u8, out[q * len ..][0..ql], want[0..ql])) return fail(cfg.rank, "variable all-gather", k, len);
            }
            trades += 1;
        }
    }
    try c.check();
    std.debug.print("PASS tcp-host rank {d}/2: {d} all-gathers (1 B to 4 MiB), {d} fp32 and bf16 sums, {d} exchanges, send/recv and variable all-gathers bit-exact over {s}\n", .{ cfg.rank, gathers, sums, trades, link });
    // 2. one call's host-to-host time by size: median and p90 of back-to-back calls; gb_s_p50 is shard bytes over p50
    var times: [500]u64 = undefined;
    for (sizes) |len| {
        inline for (.{ "all_gather", "exchange" }) |what| {
            for (&times) |*t| {
                const t0 = tp.sock.nowNs();
                if (comptime std.mem.eql(u8, what, "all_gather")) {
                    try c.allGather(@intFromPtr(in.ptr), @intFromPtr(out.ptr), len, .u8, null);
                } else try c.exchange(@intFromPtr(in.ptr), @intFromPtr(out.ptr), len, .u8, peer, null);
                t.* = tp.sock.nowNs() - t0;
            }
            std.mem.sort(u64, &times, {}, std.sort.asc(u64));
            if (cfg.rank == 0) std.debug.print("RESULT {{\"backend\":\"tcp-host\",\"link\":\"{s}\",\"world\":2,\"call\":\"{s}\",\"bytes\":{d},\"p50_us\":{d:.2},\"p90_us\":{d:.2},\"gb_s_p50\":{d:.2}}}\n", .{
                link,                                                                        what, len, @as(f64, @floatFromInt(times[250])) / 1000.0, @as(f64, @floatFromInt(times[450])) / 1000.0,
                @as(f64, @floatFromInt(len)) / @as(f64, @floatFromInt(@max(times[250], 1))),
            });
        }
    }
    try c.check();
    try b.agree("done", "tcp-host end");
    return 0;
}

fn fail(rank: u32, what: []const u8, k: usize, len: usize) u8 {
    std.debug.print("FAIL tcp-host rank {d}: {s} {d} of {d} bytes differs from the reference\n", .{ rank, what, k, len });
    return 1;
}

/// The link's name with quotes, backslashes and control bytes replaced, so a RESULT line stays JSON.
fn jsonSafe(s: []const u8, buf: []u8) []const u8 {
    const n = @min(s.len, buf.len);
    for (s[0..n], buf[0..n]) |ch, *o| o.* = if (ch == '"' or ch == '\\' or ch < 0x20) '?' else ch;
    return buf[0..n];
}
