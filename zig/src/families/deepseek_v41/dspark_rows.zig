//! Host side of DSpark passes over several slots: grouping, pass statics, plan-link messages, candidate order.

const std = @import("std");
const emit = @import("dspark_emit.zig");

/// Plan-link op 40 (every rank, leader order): [op, slot, start, src, skip, n]: ingest n taps rows; src -1: restart.
pub const op_ingest: i64 = 40;
/// 41: [op, k, (slot, anchor, start) x k]: one pass over k slots (their rows in this order).
pub const op_propose: i64 = 41;
/// 42: [op, slot]: the slot's draft context restarts (a new request).
pub const op_reset: i64 = 42;

pub const Error = error{ BadPlan, Group, Rows };

/// TF_DSV41_SLOT_DRAFTS=1: DSpark drafts with several live slots (default off: drafts refused with TF_DSV41_SLOTS > 1).
pub fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_SLOT_DRAFTS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The most slots a pass: rows below 16, or within TF_DSV41_ROWS_CAP past 16; TF_DSV41_SLOT_DRAFT_GROUP lowers it.
pub fn groupFromEnv(block: u32) !u32 {
    const cap: u32 = if (std.c.getenv("TF_DSV41_ROWS_CAP")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 16;
    // past 16 rows: every live slot (TF_DSV41_SLOTS) within the cap, in one pass
    const live: u32 = if (std.c.getenv("TF_DSV41_SLOTS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 1;
    const most = if (cap > 16) @max(1, @min(live, cap / block)) else maxGroup(block);
    const v = std.c.getenv("TF_DSV41_SLOT_DRAFT_GROUP") orelse return most;
    const g = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (g == 0 or g > most) return error.Group;
    return g;
}

pub fn maxGroup(block: u32) u32 {
    return @max(1, 15 / block);
}

/// Splits `n` asks (sorted by slot) into balanced consecutive groups of at most `most`: [a, b) each; the count.
pub fn groups(n: usize, most: usize, out: [][2]usize) usize {
    if (n == 0) return 0;
    const k = std.math.divCeil(usize, n, most) catch unreachable;
    const base = n / k;
    const extra = n % k;
    var a: usize = 0;
    for (out[0..k], 0..) |*g, i| {
        const len = base + @intFromBool(i < extra);
        g.* = .{ a, a + len };
        a += len;
    }
    return k;
}

/// One slot of a pass: its ring slot, the pending token at `start`, and the first position whose ring rows are its own.
pub const Member = struct { slot: u32, anchor: u32, start: u64, valid: u64 };

/// The pass's statics into the host mirrors in row mode; writes only the arrays the launches read, the rest zero.
pub fn stage(lay: emit.Layout, n: u32, window: u32, ring: u32, noise: u32, ms: []const Member, h64: []i64, h32: []i32) void {
    @memset(h64[0..@intCast(lay.len64)], 0);
    @memset(h32[0..@intCast(lay.len32)], 0);
    const W: usize = window;
    for (ms, 0..) |m, s| {
        const a = s * n;
        const ids = h64[@intCast(lay.ids)..][a..][0..n];
        @memset(ids, noise);
        ids[0] = m.anchor;
        for (h64[@intCast(lay.positions)..][a..][0..n], 0..) |*p, i| p.* = @intCast(m.start + i);
        @memset(h64[@intCast(lay.read)..][a..][0..n], m.slot);
        @memset(h64[@intCast(lay.write)..][a..][0..n], m.slot);
        const c0 = @max(m.start -| window, m.valid);
        const count: usize = @intCast(if (m.start > c0) m.start - c0 else 0);
        // one row's list, then copied to the slot's other rows (each row sees the same context)
        const first = h32[@intCast(lay.tokens)..][a * W ..][0..W];
        @memset(first, -1);
        for (first[0..count], 0..) |*t, j| t.* = @intCast(@as(u64, m.slot) * ring + (c0 + j) % ring);
        for (1..n) |i| @memcpy(h32[@intCast(lay.tokens)..][(a + i) * W ..][0..W], first);
        @memset(h32[@intCast(lay.counts)..][a..][0..n], @intCast(count));
        @memset(h32[@intCast(lay.lo)..][a..][0..n], @intCast(m.start));
        @memset(h32[@intCast(lay.hi)..][a..][0..n], @intCast(m.start + n - 1));
    }
}

/// The one-slot pass's statics: ids, positions, the context list as ring rows, counts, lo, hi, start, anchor.
pub fn stageOne(st: emit.Statics, ring: u64, noise: u32, valid: u64, anchor: u32, start: u64, h64: []i64, h32: []i32) void {
    const n: usize = @intCast(st.n);
    const W: usize = @intCast(st.window);
    @memset(h64, 0);
    @memset(h32, 0);
    for (0..n) |i| {
        h64[@intCast(st.ids() + @as(i64, @intCast(i)))] = if (i == 0) anchor else noise;
        h64[@intCast(st.positions() + @as(i64, @intCast(i)))] = @intCast(start + i);
    }
    const first = @max(start -| @as(u64, W), valid);
    const count: usize = @intCast(if (start > first) start - first else 0);
    h32[@intCast(st.start())] = @intCast(start);
    const tok = h32[@intCast(st.tokens())..][0 .. n * W];
    @memset(tok, -1);
    for (0..n) |i| for (0..count) |j| {
        tok[i * W + j] = @intCast((first + j) % ring);
    };
    @memset(h32[@intCast(st.counts())..][0..n], @intCast(count));
    @memset(h32[@intCast(st.lo())..][0..n], @intCast(start));
    @memset(h32[@intCast(st.hi())..][0..n], @intCast(start + n - 1));
    h32[@intCast(st.anchor())] = @intCast(anchor);
}

/// glue.cu ds_stage's arithmetic on the host (tests `stageOne`): position-dependent statics from a device pick.
pub fn stageMirror(st: emit.Statics, ring: u64, noise: u32, valid: u64, pick: []const i64, nw: usize, h64: []i64, h32: []i32) void {
    const n: usize = @intCast(st.n);
    const W: i64 = st.window;
    const A = pick[nw + 1];
    const P = pick[nw + 2];
    const lo_w: i64 = if (P > W) P - W else 0;
    const first: i64 = @max(lo_w, @as(i64, @intCast(valid)));
    const count: i64 = if (P > first) P - first else 0;
    for (0..n) |i| {
        const ii: i64 = @intCast(i);
        h64[@intCast(ii)] = if (i == 0) A else noise;
        h64[@intCast(st.positions() + ii)] = P + ii;
        h32[@intCast(st.counts() + ii)] = @intCast(count);
        h32[@intCast(st.lo() + ii)] = @intCast(P);
        h32[@intCast(st.hi() + ii)] = @intCast(P + @as(i64, @intCast(n)) - 1);
    }
    for (0..n * @as(usize, @intCast(W))) |e| {
        const j: i64 = @intCast(e % @as(usize, @intCast(W)));
        h32[@intCast(st.tokens() + @as(i64, @intCast(e)))] = if (j < count) @intCast(@mod(first + j, @as(i64, @intCast(ring)))) else -1;
    }
    h32[0] = @intCast(P);
    h32[@intCast(st.anchor())] = @intCast(A);
}

/// glue.cu ds_accept_rows on the host (its mirror): each slot's (accepted, bonus, next position) of a row window.
pub fn acceptMirror(picks: []const i64, ids: []const i64, segs: []const [3]i64, acc: [][3]i64) void {
    for (segs, acc) |sg, *o| {
        const a: usize = @intCast(sg[0]);
        const n: usize = @intCast(sg[1]);
        var c: usize = 0;
        while (c + 1 < n and ids[a + c + 1] == picks[a + c]) c += 1;
        o.* = .{ @intCast(c), picks[a + c], sg[2] + @as(i64, @intCast(c)) + 1 };
    }
}

/// glue.cu ds_stage_rows on the host (its mirror): a pass's members' position-dependent statics from acc.
pub fn stageRowsMirror(lay: emit.Layout, n: u32, window: u32, ring: u32, noise: u32, acc: []const [3]i64, mem: []const [3]i64, h64: []i64, h32: []i32) void {
    const W: i64 = window;
    for (mem, 0..) |m, mi| {
        const A = acc[@intCast(m[0])][1];
        const P = acc[@intCast(m[0])][2];
        const lo_w: i64 = if (P > W) P - W else 0;
        const first: i64 = @max(lo_w, m[2]);
        const count: i64 = if (P > first) P - first else 0;
        const a: i64 = @as(i64, @intCast(mi)) * n;
        for (0..n) |i| {
            const ii: i64 = @intCast(i);
            h64[@intCast(lay.ids + a + ii)] = if (i == 0) A else noise;
            h64[@intCast(lay.positions + a + ii)] = P + ii;
            h32[@intCast(lay.counts + a + ii)] = @intCast(count);
            h32[@intCast(lay.lo + a + ii)] = @intCast(P);
            h32[@intCast(lay.hi + a + ii)] = @intCast(P + n - 1);
        }
        for (0..@as(usize, n) * @as(usize, window)) |e| {
            const j: i64 = @intCast(e % window);
            h32[@intCast(lay.tokens + a * W + @as(i64, @intCast(e)))] = if (j < count) @intCast(m[1] * ring + @mod(first + j, ring)) else -1;
        }
    }
}

/// The leader's op_propose message for `ms` into `buf` (2 + 3 x 16 words).
pub fn proposeMsg(ms: []const Member, buf: []i64) ![]const i64 {
    if (ms.len == 0 or 2 + 3 * ms.len > buf.len) return error.BadPlan;
    buf[0] = op_propose;
    buf[1] = @intCast(ms.len);
    for (ms, 0..) |m, i| {
        buf[2 + 3 * i] = m.slot;
        buf[3 + 3 * i] = m.anchor;
        buf[4 + 3 * i] = @intCast(m.start);
    }
    return buf[0 .. 2 + 3 * ms.len];
}

/// A follower's members from op_propose (`valid` its own per-slot context starts).
pub fn proposeOf(msg: []const i64, valid: []const u64, out: []Member) ![]Member {
    if (msg.len < 2 or msg[0] != op_propose) return error.BadPlan;
    const k: usize = @intCast(msg[1]);
    if (k == 0 or k > out.len or msg.len != 2 + 3 * k) return error.BadPlan;
    for (out[0..k], 0..) |*m, i| {
        const slot: u32 = @intCast(msg[2 + 3 * i]);
        if (slot >= valid.len) return error.BadPlan;
        m.* = .{ .slot = slot, .anchor = @intCast(msg[3 + 3 * i]), .start = @intCast(msg[4 + 3 * i]), .valid = valid[slot] };
    }
    return out[0..k];
}

/// Where an ingest's taps rows are: `src` (0 "w.taps", 1 the stash, -1 not on the device) and the row within it.
pub const TapsAt = struct { src: i64, skip: i64 };

/// A slot's taps rows `row0 .. row0 + n` in "w.taps" or the stash; rows not on the device restart the context.
pub fn locate(address: u64, rows: u32, stride: u64, row0: u32, n: u32, ingest_rows: u32, taps_base: u64, stash: [2]u64) !TapsAt {
    if (row0 + n > rows or n > ingest_rows) return .{ .src = -1, .skip = 0 };
    const in_stash = address >= stash[0] and address < stash[0] + stash[1];
    const base = if (in_stash) stash[0] else taps_base;
    if (address < base or (address - base) % stride != 0) return error.Rows;
    return .{ .src = @intFromBool(in_stash), .skip = @intCast((address - base) / stride + row0) };
}

/// The ranks' gathered candidates [W][R][2k] (values, then ids as fp32 bits) into rows of [W k]: `cand` ids, `cval`.
pub fn ungather(all: []const f32, world: usize, R: usize, k: usize, cand: []i32, cval: []f32) void {
    const c = world * k;
    for (0..world) |w| for (0..R) |i| {
        const src = all[(w * R + i) * 2 * k ..][0 .. 2 * k];
        @memcpy(cval[i * c + w * k ..][0..k], src[0..k]);
        for (src[k..], cand[i * c + w * k ..][0..k]) |v, *o| o.* = @bitCast(v);
    };
}
