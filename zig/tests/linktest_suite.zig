//! The link measurements over Thunderbolt's two-sided UC: opcodes, ping-pong, bandwidth, integrity, barrier, all-reduce.
const std = @import("std");
const fabric = @import("fabric");
const collective = fabric.collective;
const vqp = fabric.verbs_qp;
const vl = fabric.verbs_link;
const sr = fabric.sendrecv;
const now = fabric.words.nowNs;

const mib = 1 << 20;
const max_row = 14336;
pub const ex_ring = std.mem.alignForward(usize, 3 * ((256 * max_row + 1 + vl.packet - 1) / vl.packet) * vl.packet, 16384);
pub const ex_stage = 4 * mib;
pub const ex_tx = 8 * mib;
pub const raw_ring = 4064 * vl.packet;
pub const raw_stage = 16 * mib;
pub const source_bytes = 32 * mib;

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    rank: u32,
    raw: *vqp.Endpoint,
    ring: *vl.Ring,
    source: *vl.Region,
    x: sr.Exchange,
    quick: bool,
    scratch: [2][]const u8 = undefined,

    fn sync(c: *Ctx) !void {
        try collective.barrier(&c.x, &c.scratch);
    }

    fn reps(c: *const Ctx, full: usize) usize {
        return if (c.quick) @max(full / 20, 5) else full;
    }

    fn send(c: *Ctx, offset: usize, len: usize) !void {
        try c.raw.send(c.source.mem[offset..][0..len], c.source.mr.lkey, offset);
    }

    /// The next whole message on the measurement queue pair, waiting up to ten seconds.
    fn take(c: *Ctx) ![]const u8 {
        const deadline = now() + 10 * std.time.ns_per_s;
        while (true) {
            if (try c.ring.next()) |m| return m;
            if (now() > deadline) return error.PeerTimeout;
            if (c.raw.outstanding > 0) try c.raw.reapSends();
        }
    }
};

/// One output line, written whole to stdout.
pub fn out(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    _ = std.c.write(1, line.ptr, line.len);
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

/// Percentiles of `samples` (sorted in place) in microseconds, after the tags.
fn report(comptime what: []const u8, comptime tags: []const u8, args: anytype, samples: []u64) void {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const s = samples;
    const at = struct {
        fn f(v: []const u64, q: f64) u64 {
            return v[@min(v.len - 1, @as(usize, @intFromFloat(q * @as(f64, @floatFromInt(v.len)))))];
        }
    }.f;
    var buf: [256]u8 = undefined;
    const t = std.fmt.bufPrint(&buf, tags, args) catch "";
    out("R " ++ what ++ " {s} n={d} min_us={d:.3} p10_us={d:.3} p50_us={d:.3} p90_us={d:.3} p99_us={d:.3} max_us={d:.3}", .{ t, s.len, us(s[0]), us(at(s, 0.1)), us(at(s, 0.5)), us(at(s, 0.9)), us(at(s, 0.99)), us(s[s.len - 1]) });
}

pub fn all(c: *Ctx) !void {
    try c.sync();
    inline for (.{ sendCompletions, opcodes, pingpong, sendCompletion, bandwidth, integrity, barriers, allReduce }, .{ "send_cqes", "opcodes", "pingpong", "send_completion", "bandwidth", "integrity", "barrier", "allreduce" }) |phase, name| {
        phase(c) catch |err| {
            out("R error phase={s} what={s} sends_in_flight={d} packets={d} posted={d} reaped={d} duplicates={d} coalesced={d} ring_filled={d} ring_returned={d}", .{ name, @errorName(err), c.raw.outstanding, c.raw.packets, c.raw.posted, c.raw.reaped, c.raw.duplicates, c.raw.coalesced, c.ring.filled, c.ring.returned });
            return err;
        };
    }
    out("R completions rank={d} sends={d} duplicates={d} coalesced={d}", .{ c.rank, c.raw.reaped, c.raw.duplicates, c.raw.coalesced });
    try c.sync();
}

/// One SEND of each size, then every send completion that arrives for it, counted with its wr_id and status.
fn sendCompletions(c: *Ctx) !void {
    for ([_]usize{ 8, 4096, 4097, 16384, 65536, 262144, mib, 4 * mib, 8 * mib }) |size| {
        try c.sync();
        if (c.rank == 0) {
            const t0 = now();
            var sge: fabric.verbs_abi.Sge = .{ .addr = @intFromPtr(c.source.mem.ptr), .length = @intCast(size), .lkey = c.source.mr.lkey };
            var wr: fabric.verbs_abi.SendWr = .{ .wr_id = 7000 + size, .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = fabric.verbs_abi.wr_send, .send_flags = fabric.verbs_abi.send_signaled };
            var bad: ?*fabric.verbs_abi.SendWr = null;
            if (c.raw.ctx.ops.post_send(c.raw.qp, &wr, &bad) != 0) return error.PostFailed;
            var wc: [64]fabric.verbs_abi.Wc = undefined;
            var total: usize = 0;
            var first_us: f64 = 0;
            var ids: [4]u64 = @splat(0);
            while (now() - t0 < 200 * std.time.ns_per_ms) {
                const n = c.raw.ctx.ops.poll_cq(c.raw.scq, 64, &wc);
                if (n < 0) return error.Completion;
                for (wc[0..@intCast(n)]) |w| {
                    if (total == 0) first_us = us(now() - t0);
                    if (total < ids.len) ids[total] = w.wr_id;
                    if (w.status != 0) out("R send_cqe bytes={d} status={d}", .{ size, w.status });
                    total += 1;
                }
            }
            out("R send_cqes bytes={d} packets={d} completions={d} first_us={d:.1} wr_ids={d},{d},{d},{d}", .{ size, (size + 4095) / 4096, total, first_us, ids[0], ids[1], ids[2], ids[3] });
        } else {
            _ = try c.take();
            try c.ring.release();
        }
    }
    try c.sync();
}

/// Rank 0 posts a WRITE, a WRITE with immediate data and a SEND of 8 bytes each; rank 1 reports what arrived.
fn opcodes(c: *Ctx) !void {
    if (c.rank == 0) {
        for (0..3) |k| std.mem.writeInt(u64, c.source.mem[k * 64 ..][0..8], 0xC0DE0000 + k, .little);
        try c.raw.write(c.source.mem[0..8], c.source.mr.lkey, 0, 0, null, 0);
        try c.raw.write(c.source.mem[64..72], c.source.mr.lkey, 0, 0, 0x1234, 1);
        try c.raw.send(c.source.mem[128..136], c.source.mr.lkey, 2);
    } else {
        var got: usize = 0;
        const t0 = now();
        while (now() - t0 < 500 * std.time.ns_per_ms) {
            const m = (try c.ring.next()) orelse continue;
            out("R opcode message={d} bytes={d} data=0x{x}", .{ got, m.len, if (m.len >= 8) std.mem.readInt(u64, m[0..8], .little) else 0 });
            try c.ring.release();
            got += 1;
        }
        out("R opcode messages={d} of 3 posted (write, write_imm, send)", .{got});
    }
    try c.raw.drain();
    try c.sync();
}

/// SEND and answer; half the round trip is the one-way latency, the receiver's ring always posted.
fn pingpong(c: *Ctx) !void {
    const samples = try c.gpa.alloc(u64, 2000);
    defer c.gpa.free(samples);
    for ([_]usize{ 8, 64, 4096, 4097, 12288, 14336, 65536, mib, 8 * mib }) |size| {
        const n = c.reps(if (size <= 65536) 2000 else if (size <= mib) 500 else 60);
        try c.sync();
        for (0..n + 5) |i| {
            const t0 = now();
            if (c.rank == 0) {
                try c.send(0, size);
                if ((try c.take()).len != size) return error.SizeMismatch;
                try c.ring.release();
            } else {
                if ((try c.take()).len != size) return error.SizeMismatch;
                try c.ring.release();
                try c.send(0, size);
            }
            if (i >= 5) samples[i - 5] = now() - t0;
        }
        try c.raw.drain();
        if (c.rank == 0) report("pingpong_rtt", "bytes={d}", .{size}, samples[0..n]);
    }
}

/// Queue depth one: post a SEND and wait for its local completion while the peer consumes.
fn sendCompletion(c: *Ctx) !void {
    const samples = try c.gpa.alloc(u64, 500);
    defer c.gpa.free(samples);
    for ([_]usize{ 64, 4096, 12288, 65536, mib, 8 * mib }) |size| {
        const n = c.reps(if (size <= mib) 500 else 60);
        try c.sync();
        for (0..n) |i| {
            if (c.rank == 0) {
                const t0 = now();
                try c.send(0, size);
                try c.raw.drain();
                samples[i] = now() - t0;
            } else {
                _ = try c.take();
                try c.ring.release();
            }
        }
        if (c.rank == 0) report("send_completion", "bytes={d}", .{size}, samples[0..n]);
    }
}

/// Send `sends` messages and consume `recvs`, interleaved so neither side's ring can stall the other.
fn stream(c: *Ctx, size: usize, sends: usize, recvs: usize) !u64 {
    var sent: usize = 0;
    var got: usize = 0;
    var bytes: u64 = 0;
    const deadline = now() + 20 * std.time.ns_per_s;
    while (sent < sends or got < recvs or c.raw.outstanding > 0) {
        if (now() > deadline) {
            out("R stream_stalled bytes={d} sent={d} of {d} received={d} of {d}", .{ size, sent, sends, got, recvs });
            return error.PeerTimeout;
        }
        if (sent < sends and c.raw.packets + (size + vl.packet - 1) / vl.packet <= 4094) {
            try c.send(0, size);
            sent += 1;
        }
        if (c.raw.outstanding > 0) try c.raw.reapSends();
        if (got < recvs) if (try c.ring.next()) |m| {
            bytes += m.len;
            try c.ring.release();
            got += 1;
        };
    }
    return bytes;
}

/// Back-to-back messages for about two gigabytes: one way each direction, then both at once.
fn bandwidth(c: *Ctx) !void {
    for (0..3) |who| {
        for ([_]usize{ 4096, 16384, 65536, 262144, mib, 4 * mib, 8 * mib }) |size| {
            if (who == 2 and size < mib) continue;
            const total: usize = if (c.quick) 128 * mib else 2048 * mib;
            const count = @max(total / size, 8);
            const sends = if (c.rank == who or who == 2) count else 0;
            const recvs = if (c.rank != who or who == 2) count else 0;
            try c.sync();
            const t0 = now();
            const bytes = try stream(c, size, sends, recvs);
            const dt = now() - t0;
            const gbs = @as(f64, @floatFromInt(@max(bytes, sends * size))) / @as(f64, @floatFromInt(dt));
            out("R bandwidth rank={d} mode={s} bytes={d} messages={d} received={d} seconds={d:.3} GB_s={d:.3} Gb_s={d:.2}", .{ c.rank, if (who == 2) "both" else if (who == 0) "0to1" else "1to0", size, count, bytes, @as(f64, @floatFromInt(dt)) / 1e9, gbs, gbs * 8 });
        }
    }
}

fn fill(buf: []u8, seq: u64) void {
    for (std.mem.bytesAsSlice(u64, buf), 0..) |*w, i| w.* = seq *% 0x9E3779B97F4A7C15 ^ i *% 0xC2B2AE3D27D4EB4F;
}

fn wrong(buf: []const u8, seq: u64) usize {
    var bad: usize = 0;
    for (std.mem.bytesAsSlice(u64, buf), 0..) |w, i| bad += @intFromBool(w != seq *% 0x9E3779B97F4A7C15 ^ i *% 0xC2B2AE3D27D4EB4F);
    return bad;
}

/// Every word of every message checked as it streams, rank 0 to rank 1, wrapped messages through the stage.
fn integrity(c: *Ctx) !void {
    for ([_][2]usize{ .{ 12288, 20000 }, .{ mib, 2000 }, .{ 8 * mib, 200 } }) |sc| {
        const size = sc[0];
        const total = c.reps(sc[1]);
        const slots = source_bytes / size;
        try c.sync();
        const t0 = now();
        var bad: usize = 0;
        var sizes: usize = 0;
        for (0..total) |k| {
            if (c.rank == 0) {
                while (c.raw.outstanding >= slots - 1 or c.raw.packets + size / vl.packet > 4094) try c.raw.reapSends();
                const at = (k % slots) * size;
                fill(c.source.mem[at..][0..size], k);
                try c.send(at, size);
            } else {
                const m = try c.take();
                if (m.len != size) sizes += 1 else bad += wrong(m, k);
                try c.ring.release();
            }
        }
        try c.raw.drain();
        if (c.rank == 1) out("R integrity bytes={d} messages={d} wrong_size={d} wrong_words={d} seconds={d:.3}", .{ size, total, sizes, bad, @as(f64, @floatFromInt(now() - t0)) / 1e9 });
    }
}

/// Our two-sided barrier, back to back.
fn barriers(c: *Ctx) !void {
    const n = c.reps(5000);
    const samples = try c.gpa.alloc(u64, n);
    defer c.gpa.free(samples);
    try c.sync();
    for (samples) |*s| {
        const t0 = now();
        try c.sync();
        s.* = now() - t0;
    }
    report("barrier", "rank={d}", .{c.rank}, samples);
}

/// One row's bf16 partial on one rank, a function of (rank, row id) only.
fn partialRow(rank: u32, id: u64, row: []u8) void {
    var prng: std.Random.DefaultPrng = .init(id *% 1_000_003 +% rank);
    const r = prng.random();
    for (0..row.len / 2) |i| {
        const v = (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 6)));
        std.mem.writeInt(u16, row[2 * i ..][0..2], collective.toBf16(v), .little);
    }
}

/// Our rank-ordered all-reduce of bf16 rows; every result is checked against the local fp32 reference and hashed.
fn allReduce(c: *Ctx) !void {
    for ([_]usize{ 12288, max_row }) |row| {
        for ([_]usize{ 1, 4, 16, 64, 256 }) |rows| {
            const len = rows * row;
            const bufs = try c.gpa.alloc(u8, 4 * len);
            defer c.gpa.free(bufs);
            const mine = bufs[0..len];
            const result = bufs[len .. 2 * len];
            const parts = [2][]u8{ bufs[2 * len .. 3 * len], bufs[3 * len .. 4 * len] };
            const n = c.reps(if (rows <= 16) 500 else if (rows <= 64) 120 else 30);
            const samples = try c.gpa.alloc(u64, 2 * n);
            defer c.gpa.free(samples);
            var sha = std.crypto.hash.sha2.Sha256.init(.{});
            var bad: usize = 0;
            for (0..n) |it| {
                for (0..rows) |j| for (0..2) |r| partialRow(@intCast(r), (it * rows + j) * 7 + row, parts[r][j * row ..][0..row]);
                @memcpy(mine, parts[c.rank]);
                try c.sync();
                const t0 = now();
                try collective.allReduce(&c.x, .bf16, mine, result, &c.scratch);
                samples[it] = now() - t0;
                const t1 = now();
                collective.reduce(.bf16, &.{ parts[0], parts[1] }, mine);
                samples[n + it] = now() - t1;
                for (mine, result) |a, b| bad += @intFromBool(a != b);
                sha.update(result);
            }
            var digest: [32]u8 = undefined;
            sha.final(&digest);
            report("allreduce", "rank={d} rows={d} row_bytes={d} wrong_bytes={d} sha={s}", .{ c.rank, rows, row, bad, std.fmt.bytesToHex(digest, .lower)[0..16] }, samples[0..n]);
            report("cpu_sum", "rank={d} rows={d} row_bytes={d}", .{ c.rank, rows, row }, samples[n..]);
        }
    }
}
