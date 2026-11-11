//! Mixed-TP rules 1 and 2: every collective equals the host reference, and two runs match byte for byte.
const std = @import("std");
const dump = @import("tp_check_dump.zig");
const tp = @import("tp");
const reference = tp.reference;
const DType = tp.collective.DType;
const Error = dump.Error;
const Collective = dump.Collective;
const Dump = dump.Dump;
pub const Mismatch = struct {
    seq: u64,
    layer: u32,
    rank: u32,
    /// What went wrong; for a value mismatch, the first differing element and both values' bits.
    why: []const u8,
    element: usize = 0,
    got: u64 = 0,
    want: u64 = 0,
};

pub const ReferenceReport = struct {
    calls: usize = 0,
    /// Every recv buffer checked, one per rank per call.
    buffers: usize = 0,
    first: ?Mismatch = null,
    mismatches: usize = 0,
};

fn elemBits(b: []const u8, t: DType, i: usize) u64 {
    const s = t.size();
    return switch (s) {
        1 => b[i],
        2 => std.mem.readInt(u16, b[2 * i ..][0..2], .little),
        4 => std.mem.readInt(u32, b[4 * i ..][0..4], .little),
        else => std.mem.readInt(u64, b[8 * i ..][0..8], .little),
    };
}

fn note(r: *ReferenceReport, m: Mismatch) void {
    r.mismatches += 1;
    if (r.first == null) r.first = m;
}

/// Compare `got` with `want` bit for bit; on a difference, record the first differing element.
fn compare(r: *ReferenceReport, c: Collective, rank: u32, got: []const u8, want: []const u8) void {
    r.buffers += 1;
    if (got.len != want.len) return note(r, .{ .seq = c.seq, .layer = c.layer, .rank = rank, .why = "length", .got = got.len, .want = want.len });
    if (std.mem.eql(u8, got, want)) return;
    const n = want.len / c.dtype.size();
    for (0..n) |i| {
        const g = elemBits(got, c.dtype, i);
        const w = elemBits(want, c.dtype, i);
        if (g != w) return note(r, .{ .seq = c.seq, .layer = c.layer, .rank = rank, .why = "value", .element = i, .got = g, .want = w });
    }
}

/// Rule 1: dumps[r] is rank r's; the n-th calls must match, and each recv must equal the rank-order host reference.
pub fn checkReference(a: std.mem.Allocator, dumps: []const Dump) Error!ReferenceReport {
    var r: ReferenceReport = .{};
    if (dumps.len == 0) return r;
    for (dumps, 0..) |d, i| {
        if (d.header.rank != i or d.header.world != dumps.len) {
            note(&r, .{ .seq = 0, .layer = 0, .rank = @intCast(i), .why = "rank or world in the header" });
            return r;
        }
        if (d.collectives.len != dumps[0].collectives.len) {
            note(&r, .{ .seq = 0, .layer = 0, .rank = @intCast(i), .why = "call count", .got = d.collectives.len, .want = dumps[0].collectives.len });
            return r;
        }
    }
    const sends = try a.alloc([]const u8, dumps.len);
    defer a.free(sends);
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(a);
    for (0..dumps[0].collectives.len) |k| {
        const c0 = dumps[0].collectives[k];
        r.calls += 1;
        var same = true;
        for (dumps, 0..) |d, i| {
            const c = d.collectives[k];
            sends[i] = c.send;
            if (c.seq != c0.seq or c.layer != c0.layer or c.phase != c0.phase or c.call != c0.call or c.dtype != c0.dtype or c.op != c0.op or c.send.len != c0.send.len) {
                note(&r, .{ .seq = c0.seq, .layer = c0.layer, .rank = @intCast(i), .why = "ranks disagree on the call" });
                same = false;
                break;
            }
        }
        if (!same) continue;
        switch (c0.call) {
            .all_reduce => {
                try want.resize(a, c0.send.len);
                reference.reduce(want.items, sends, c0.dtype, c0.op);
                for (dumps, 0..) |d, i| compare(&r, d.collectives[k], @intCast(i), d.collectives[k].recv, want.items);
            },
            .all_gather => {
                want.clearRetainingCapacity();
                for (sends) |s| try want.appendSlice(a, s);
                for (dumps, 0..) |d, i| compare(&r, d.collectives[k], @intCast(i), d.collectives[k].recv, want.items);
            },
            .exchange => for (dumps, 0..) |d, i| {
                const c = d.collectives[k];
                if (c.peer >= dumps.len) {
                    note(&r, .{ .seq = c.seq, .layer = c.layer, .rank = @intCast(i), .why = "peer out of range", .got = c.peer });
                    continue;
                }
                compare(&r, c, @intCast(i), c.recv, dumps[c.peer].collectives[k].send);
            },
        }
    }
    return r;
}

pub const RepeatReport = struct {
    records: usize = 0,
    /// Index (in record order) of the first collective or logits record that differs, if any.
    first_collective: ?usize = null,
    first_logits: ?usize = null,
    same_shape: bool = true,
};

/// Rule 2: two dumps of the same rank from two runs. Every record must match byte for byte (sends, recvs, logits).
pub fn checkRepeat(x: Dump, y: Dump) RepeatReport {
    var r: RepeatReport = .{};
    if (x.collectives.len != y.collectives.len or x.logits.len != y.logits.len) r.same_shape = false;
    for (0..@min(x.collectives.len, y.collectives.len)) |i| {
        r.records += 1;
        const p = x.collectives[i];
        const q = y.collectives[i];
        const same = p.seq == q.seq and p.layer == q.layer and p.phase == q.phase and p.call == q.call and p.dtype == q.dtype and p.op == q.op and
            p.peer == q.peer and std.mem.eql(u8, p.send, q.send) and std.mem.eql(u8, p.recv, q.recv);
        if (!same) {
            r.first_collective = i;
            break;
        }
    }
    for (0..@min(x.logits.len, y.logits.len)) |i| {
        r.records += 1;
        const p = x.logits[i];
        const q = y.logits[i];
        if (p.prompt != q.prompt or p.position != q.position or p.chosen != q.chosen or !std.mem.eql(u8, p.bytes, q.bytes)) {
            r.first_logits = i;
            break;
        }
    }
    return r;
}

pub fn repeatPassed(r: RepeatReport) bool {
    return r.records > 0 and r.same_shape and r.first_collective == null and r.first_logits == null;
}
