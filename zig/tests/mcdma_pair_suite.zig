//! tf-mcdma-pair's suite: each test's steps on one rank, its result buffers hashed beside the host reference.
const std = @import("std");
const fabric = @import("fabric");
const pair = @import("mcdma_pair_rank.zig");
const collective = fabric.collective;
const words = fabric.words;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Rank = pair.Rank;
const Phase = pair.Phase;
const mib = pair.mib;
const hidden = pair.hidden;
const max_rows = pair.max_rows;
const pipe_slot = pair.pipe_slot;
const reduce_steps = pair.reduce_steps;

pub fn run(r: *Rank, t: Phase, tx: *Tx) !void {
    const rs = r.s.seed +% @as(u64, r.round) *% 0x1000193;
    switch (t) {
        .barrier => for (0..3) |s| {
            try r.step(t, s);
            try collective.barrier(&r.plan_ch, &r.scratch);
        },
        .broadcast => for (0..20) |s| {
            try r.step(t, s);
            const root: u32 = @intCast(s % 2);
            var plan: [24]u8 = undefined;
            for (&plan, 0..) |*b, i| b.* = pattern(rs, root, s, i);
            const got = try collective.broadcast(&r.plan_ch, root, if (r.rank == root) &plan else "", &r.scratch);
            try tx.add("s{d}:root{d}", .{ s, root }, got, &plan);
        },
        .allreduce_bf16, .allreduce_f32 => try allReduce(r, t, rs, tx),
        .alltoall => for (0..30) |s| {
            try r.step(t, s);
            var bufs: [2][64 * 9]u8 = undefined;
            var chunks: [2][]const u8 = undefined;
            for (&chunks, &bufs, 0..) |*c, *b, p| {
                const n = (r.rank * 7 + p * 3 + s) % 9 * 64;
                for (b[0..n], 0..) |*x, i| x.* = pattern(rs, r.rank, s * 2 + p, i);
                c.* = b[0..n];
            }
            try collective.allToAll(&r.ch, &chunks, &r.scratch);
            for (0..2) |src| {
                const dst = 1 - src;
                const n = (src * 7 + dst * 3 + s) % 9 * 64;
                if (src == r.rank) {
                    try tx.add("s{d}:{d}>{d}", .{ s, src, dst }, chunks[dst], chunks[dst]);
                    continue;
                }
                if (r.scratch[src].len != n) return mismatch(r, error.SizeMismatch, t, s, r.scratch[src].len, n);
                for (r.want[0..n], 0..) |*x, i| x.* = pattern(rs, @intCast(src), s * 2 + dst, i);
                try tx.add("s{d}:{d}>{d}", .{ s, src, dst }, r.scratch[src], r.want[0..n]);
            }
        },
        .gather => for (0..30) |s| {
            try r.step(t, s);
            const root: u32 = @intCast(s % 2);
            const src: u32 = 1 - root;
            var mine: [40]u8 = undefined;
            for (&mine, 0..) |*b, i| b.* = pattern(rs, r.rank, s + 500, i);
            try collective.gather(&r.ch, root, &mine, &r.scratch);
            if (r.rank != root) {
                try tx.add("s{d}:{d}>{d}", .{ s, src, root }, &mine, &mine);
                continue;
            }
            var want: [40]u8 = undefined;
            for (&want, 0..) |*b, i| b.* = pattern(rs, src, s + 500, i);
            if (r.scratch[src].len != mine.len) return mismatch(r, error.SizeMismatch, t, s, r.scratch[src].len, mine.len);
            try tx.add("s{d}:{d}>{d}", .{ s, src, root }, r.scratch[src], &want);
        },
        .pipes => for (0..100) |k| {
            try r.step(t, k);
            const msg = r.big[0..pipe_slot];
            const n = (k * 977 + r.rank * 13) % pipe_slot;
            for (msg[0..n], 0..) |*b, i| b.* = pattern(rs, r.rank, k + 2000, i);
            try r.tx.send(msg[0..n]);
            const m = try r.rx.recv();
            const pn = (k * 977 + r.peer * 13) % pipe_slot;
            if (m.len != pn) return mismatch(r, error.SizeMismatch, t, k, m.len, pn);
            for (r.want[0..pn], 0..) |*b, i| b.* = pattern(rs, r.peer, k + 2000, i);
            try tx.both("k{d}", .{k}, msg[0..n], msg[0..n], m, r.want[0..pn]);
            try r.rx.release();
        },
        .bulk => {
            const rd = r.rdma();
            const l = r.lay;
            try r.step(t, 0);
            for (r.big, 0..) |*b, i| b.* = pattern(rs, r.rank, 77, i);
            const done = tag(r.inv, 1);
            const t0 = words.nowNs(); // the wire alone: this rank's writes landing, and the peer's into this window
            try collective.bulk(rd, r.peer, l.bulk_at, r.big, 2 * mib, l.flags_at + 8, done);
            try r.step(t, 1);
            try waitFor(r, l.flags_at + 8, done);
            const ns = words.nowNs() - t0;
            const gbit = @as(f64, @floatFromInt(l.bulk_len)) * 8 / @as(f64, @floatFromInt(@max(ns, 1)));
            r.out.line("WIRE test=bulk round={d} bytes_each_way={d} s={d:.4} gbit_each_way={d:.2}", .{ r.round, l.bulk_len, @as(f64, @floatFromInt(ns)) / 1e9, gbit });
            for (r.big_want, 0..) |*b, i| b.* = pattern(rs, r.peer, 77, i);
            try tx.both("bulk", .{}, r.big, r.big, rd.window()[l.bulk_at..][0..l.bulk_len], r.big_want);
        },
        .signals => {
            const rd = r.rdma();
            const at = r.lay.flags_at + 16;
            const n = r.s.signals;
            // every read of the word the peer's storm lands in is the last storm's value or this test's tags
            var watch: SignalWatch = .{ .at = at, .inv = r.inv, .n = n, .old = r.signals_last };
            try r.step(t, 0);
            for (1..n + 1) |i| {
                if (i % 1000 == 0) try r.step(t, i);
                try rd.signal(r.peer, at, tag(r.inv, i));
                try watch.check(r, rd.local(at));
            }
            try rd.flush();
            try r.step(t, n + 1);
            const last = tag(r.inv, n);
            try waitSignals(r, &watch, last);
            r.signals_last = last;
            var sent: [8]u8 = undefined;
            var got: [8]u8 = undefined;
            std.mem.writeInt(u64, &sent, last, .little);
            std.mem.writeInt(u64, &got, rd.local(at), .little);
            try tx.both("last", .{}, &sent, &sent, &got, &sent);
        },
        .write2signal => {
            const rd = r.rdma();
            const l = r.lay;
            const msg = r.big[0 .. 200 + 50000];
            for (1..41) |k| {
                try r.step(t, k);
                const head_len = 1 + k * 13 % 200;
                const body_len = k * 977 % 50000;
                const n = head_len + body_len;
                for (msg[0..n], 0..) |*b, i| b.* = pattern(rs, r.rank, k + 3000, i);
                try rd.write2Signal(r.peer, l.ws_at, msg[0..head_len], msg[head_len..][0..body_len], l.flags_at + 24, tag(r.inv, k));
                try waitFor(r, l.flags_at + 24, tag(r.inv, k));
                for (r.want[0..n], 0..) |*b, i| b.* = pattern(rs, r.peer, k + 3000, i);
                try tx.both("k{d}", .{k}, msg[0..n], msg[0..n], rd.window()[l.ws_at..][0..n], r.want[0..n]);
                try collective.barrier(&r.plan_ch, &r.scratch); // the peer has it: the landing place may change
            }
        },
        .writesignalfrom => {
            const rd = r.rdma();
            const l = r.lay;
            const win = rd.window();
            const land = l.ws_at + (256 << 10);
            for (1..41) |k| {
                try r.step(t, k);
                const len = 1 + k * 2311 % 100000;
                for (win[l.src_at..][0..len], 0..) |*b, i| b.* = pattern(rs, r.rank, k + 4000, i);
                try r.ep.?.writeSignalFrom(r.peer, l.src_at, land, len, l.flags_at + 32, tag(r.inv, k));
                try waitFor(r, l.flags_at + 32, tag(r.inv, k));
                for (r.want[0..len], 0..) |*b, i| b.* = pattern(rs, r.peer, k + 4000, i);
                try tx.both("k{d}", .{k}, win[l.src_at..][0..len], win[l.src_at..][0..len], win[land..][0..len], r.want[0..len]);
                try collective.barrier(&r.plan_ch, &r.scratch); // both have theirs: sources and landing places may change
            }
            try rd.flush();
        },
        else => unreachable,
    }
}

fn allReduce(r: *Rank, t: Phase, rs: u64, tx: *Tx) !void {
    const elem: collective.Elem = if (t == .allreduce_bf16) .bf16 else .f32;
    const row = hidden * elem.size();
    var ids: [max_rows]u64 = undefined;
    for (0..reduce_steps) |s| {
        try r.step(t, s);
        const rows = rowsOf(rs, s, &ids);
        for (rows, 0..) |id, j| partialRow(rs, r.rank, id, elem, r.partial[j * row ..][0..row]);
        const len = rows.len * row;
        try collective.allReduce(&r.ch, elem, r.partial[0..len], r.result[0..len], &r.scratch);
        var parts: [2][4 * hidden]u8 = undefined;
        var solo: [4 * hidden]u8 = undefined;
        for (rows, 0..) |id, j| {
            for (&parts, 0..) |*p, pr| partialRow(rs, @intCast(pr), id, elem, p[0..row]);
            const views = [2][]const u8{ parts[0][0..row], parts[1][0..row] };
            collective.reduce(elem, &views, solo[0..row]);
            const got = r.result[j * row ..][0..row];
            if (!std.mem.eql(u8, solo[0..row], got)) {
                r.out.line("MISMATCH test={s} round={d} step={d} row={d} id={d} against=collective.reduce", .{ @tagName(t), r.round, s, j, id });
                return error.ReduceMismatch;
            }
            reference(elem, views, r.want[j * row ..][0..row]);
        }
        try tx.add("s{d}:rows{d}", .{ s, rows.len }, r.result[0..len], r.want[0..len]);
    }
}

fn waitFor(r: *Rank, at: usize, want: u64) !void {
    const rd = r.rdma();
    const deadline = words.nowNs() + @as(u64, r.s.wait_s) * std.time.ns_per_s;
    while (rd.local(at) != want) {
        if (words.nowNs() > deadline) return error.PeerTimeout;
        std.atomic.spinLoopHint();
    }
}

/// waitFor for the signal storm's word, checking every value it reads.
fn waitSignals(r: *Rank, w: *SignalWatch, want: u64) !void {
    const rd = r.rdma();
    const deadline = words.nowNs() + @as(u64, r.s.wait_s) * std.time.ns_per_s;
    while (true) {
        const v = rd.local(w.at);
        try w.check(r, v);
        if (v == want) return;
        if (words.nowNs() > deadline) return error.PeerTimeout;
        std.atomic.spinLoopHint();
    }
}

fn mismatch(r: *Rank, err: anyerror, t: Phase, s: u64, got: usize, want: usize) anyerror {
    r.out.line("MISMATCH test={s} round={d} step={d} len={d} want_len={d}", .{ @tagName(t), r.round, s, got, want });
    return err;
}

/// One test's transcript: every result buffer in the order both ranks share, and the host reference beside it.
pub const Tx = struct {
    r: *Rank,
    phase: Phase,
    got: Sha256 = .init(.{}),
    want: Sha256 = .init(.{}),
    n: usize = 0,

    fn add(tx: *Tx, comptime what: []const u8, args: anytype, got: []const u8, want: []const u8) !void {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, got.len, .little);
        tx.got.update(&len);
        tx.got.update(got);
        std.mem.writeInt(u64, &len, want.len, .little);
        tx.want.update(&len);
        tx.want.update(want);
        var label: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&label, what, args) catch "?";
        if (tx.r.s.hash_detail) {
            var one: [32]u8 = undefined;
            Sha256.hash(got, &one, .{});
            tx.r.out.line("BUF test={s} round={d} n={d} what={s} len={d} sha={s}", .{ @tagName(tx.phase), tx.r.round, tx.n, name, got.len, &std.fmt.bytesToHex(one, .lower) });
        }
        tx.n += 1;
        if (!std.mem.eql(u8, got, want)) {
            const at = std.mem.indexOfDiff(u8, got, want) orelse 0;
            tx.r.out.line("MISMATCH test={s} round={d} what={s} len={d} want_len={d} first_diff={d} got={d} want={d}", .{ @tagName(tx.phase), tx.r.round, name, got.len, want.len, at, if (at < got.len) got[at] else 0, if (at < want.len) want[at] else 0 });
            return error.ReferenceMismatch;
        }
    }

    /// One step's two directions, rank 0's bytes first: what this rank sent and what it received.
    fn both(tx: *Tx, comptime what: []const u8, args: anytype, sent: []const u8, sent_want: []const u8, recv: []const u8, recv_want: []const u8) !void {
        const me = tx.r.rank;
        var label: [56]u8 = undefined;
        const base = std.fmt.bufPrint(&label, what, args) catch "?";
        for (0..2) |src| {
            if (src == me) try tx.add("{s}:{d}>{d}", .{ base, src, 1 - src }, sent, sent_want) else try tx.add("{s}:{d}>{d}", .{ base, src, 1 - src }, recv, recv_want);
        }
    }
};

/// The storm's landing word as read: the last storm's value, then this test's tags never going back; else it fails.
const SignalWatch = struct {
    at: usize,
    inv: u64,
    n: u64,
    old: u64,
    k: u64 = 0, // the highest k read so far
    reads: u64 = 0,

    fn check(w: *SignalWatch, r: *Rank, v: u64) !void {
        w.reads += 1;
        if (v == w.old and w.k == 0) return;
        const k = v & 0xffff_ffff;
        if (v >> 32 == w.inv and k >= @max(w.k, 1) and k <= w.n) {
            w.k = k;
            return;
        }
        r.out.line("MISMATCH test=signals round={d} read={d} got=0x{x} want=0x{x} until the first signal, then 0x{x}..0x{x} never below 0x{x}", .{ r.round, w.reads, v, w.old, tag(w.inv, 1), tag(w.inv, w.n), tag(w.inv, @max(w.k, 1)) });
        return error.SignalMismatch;
    }
};

fn tag(inv: u64, k: u64) u64 {
    return inv << 32 | k;
}

fn pattern(seed: u64, rank: u32, k: u64, i: usize) u8 {
    const x: u64 = i;
    return @truncate(@as(u64, rank) *% 131 +% k *% 31 +% x *% 7 +% (x >> 9) +% seed *% 0x9E3779B1);
}

/// A row's partial depends only on (seed, rank, row id), so either rank recomputes both.
fn partialRow(seed: u64, rank: u32, id: u64, elem: collective.Elem, out: []u8) void {
    var prng: std.Random.DefaultPrng = .init(id *% 131 +% rank +% seed *% 1_000_003);
    const r = prng.random();
    for (0..hidden) |i| {
        const v = (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 6)));
        switch (elem) {
            .bf16 => std.mem.writeInt(u16, out[2 * i ..][0..2], collective.toBf16(v), .little),
            .f32 => std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(v), .little),
        }
    }
}

fn rowsOf(seed: u64, s: u64, ids: []u64) []u64 {
    const counts = [_]usize{ 1, 7, 64, 300, 2, 129, 1, 33 };
    const k = counts[s % counts.len];
    for (ids[0..k], 0..) |*id, j| id.* = (seed *% 7 +% s * 7919 +% j * 104729) % 1000;
    return ids[0..k];
}

/// The host reference: p0 + p1 in fp32, then (for bf16) round to nearest even by hand.
fn reference(elem: collective.Elem, parts: [2][]const u8, out: []u8) void {
    for (0..hidden) |i| {
        var acc: f32 = 0;
        for (parts, 0..) |p, rank| {
            const v: f32 = switch (elem) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, p[2 * i ..][0..2], .little)) << 16),
                .f32 => @bitCast(std.mem.readInt(u32, p[4 * i ..][0..4], .little)),
            };
            acc = if (rank == 0) v else acc + v;
        }
        const bits: u32 = @bitCast(acc);
        switch (elem) {
            .f32 => std.mem.writeInt(u32, out[4 * i ..][0..4], bits, .little),
            .bf16 => {
                const low = bits & 0xFFFF;
                var high = bits >> 16;
                if (low > 0x8000 or (low == 0x8000 and high & 1 == 1)) high += 1;
                std.mem.writeInt(u16, out[2 * i ..][0..2], @intCast(high & 0xFFFF), .little);
            },
        }
    }
}
