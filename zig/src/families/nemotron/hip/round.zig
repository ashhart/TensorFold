//! A lane round on a HIP rank: every stream's window in one pass over the round's device plan, each row bitwise serial.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const pages = @import("pages.zig");
const mamba = @import("mamba.zig");
const moe = @import("moe.zig");
const forward = @import("forward.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const pops = hip.plan_ops;

/// Rows a stream's window holds at most: the pending token and the drafts (every width reproduces one-token steps).
pub const max_window = 4;

/// Shortest span the attention walk covers; spans double from here (a row's bits do not depend on the span).
pub const min_span = 128;

/// A window's per-row Mamba states: slot r (r * stride bytes into `buf`) holds every layer's state after row r - 1.
pub const Snaps = struct { buf: hip.DeviceBuffer, stride: usize };

/// One stream's rows of a round: its tokens from its kept length, and where each row's Mamba state goes.
pub const Rows = struct { s: *pages.Stream, tokens: []const u32, snaps: ?Snaps = null };

/// One stream's window of a round: its rows' tokens from `pos` (the head's own position for a head round).
pub const Window = struct { s: *pages.Stream, pos: usize, tokens: []const u32 };

/// Where each array sits in the plan buffer, in 4-byte words (the 8-byte array on even words).
const Layout = struct {
    tokens: usize,
    pos: usize,
    slot: usize,
    first: usize,
    count: usize,
    desc: usize,
    words: usize,

    fn of(rows: usize, slots: usize) Layout {
        var l: Layout = undefined;
        l.tokens = 0;
        l.pos = rows;
        l.slot = 2 * rows;
        l.first = 3 * rows;
        l.count = l.first + slots;
        l.desc = std.mem.alignForward(usize, l.count + slots, 2);
        l.words = l.desc + 2 * slots;
        return l;
    }
};

/// The plan buffer of one engine, as large as its biggest round, and the pinned words a round is written in.
pub const Plan = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    rows: usize,
    slots: usize,
    /// Pages a head of a pool slot holds, and the words of a descriptor before its page table.
    pool: u32,
    pages: u32,

    /// Device bytes a plan of `rows` rows over `slots` streams holds.
    pub fn deviceBytes(rows: usize, slots: usize) usize {
        return 4 * Layout.of(rows, slots).words;
    }

    pub fn init(d: *const hip.Runtime, pool: *const pages.Pool, rows: usize, slots: usize) !Plan {
        const bytes = deviceBytes(rows, slots);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        return .{
            .host = host,
            .dev = try hip.DeviceBuffer.alloc(d, bytes),
            .rows = rows,
            .slots = slots,
            .pool = @intCast(pool.count),
            .pages = @intCast(pages.header_words + 2 * pool.keys.len),
        };
    }

    pub fn deinit(p: *Plan) void {
        p.dev.free();
        p.host.free();
    }

    /// The round's `windows` written and sent up (the copy done): the plan its launches read, and the span they walk.
    pub fn send(p: *Plan, windows: []const Window, capacity: usize, stream: hip.Stream) !Launch {
        var total: usize = 0;
        var visible: usize = 0;
        for (windows) |w| {
            total += w.tokens.len;
            visible = @max(visible, w.pos + w.tokens.len);
        }
        if (windows.len == 0 or windows.len > p.slots or total == 0 or total > p.rows) return error.WindowTooWide;
        const l = Layout.of(total, windows.len);
        const w = p.host.slice(u32);
        var at: usize = 0;
        for (windows, 0..) |win, s| {
            const desc = win.s.desc.base();
            w[l.first + s] = @intCast(at);
            w[l.count + s] = @intCast(win.tokens.len);
            w[l.desc + 2 * s] = @truncate(desc);
            w[l.desc + 2 * s + 1] = @truncate(desc >> 32);
            for (win.tokens, 0..) |t, i| {
                w[l.tokens + at + i] = t;
                w[l.pos + at + i] = @intCast(win.pos + i);
                w[l.slot + at + i] = @intCast(s);
            }
            at += win.tokens.len;
        }
        try hip.raw.upload(p.dev, 0, p.host.bytes[0 .. 4 * l.words], stream.handle);
        try stream.synchronize();
        const base = p.dev.base();
        const args: pops.Args = .{
            .pos = base + 4 * l.pos,
            .slot = base + 4 * l.slot,
            .first = base + 4 * l.first,
            .count = base + 4 * l.count,
            .desc = base + 4 * l.desc,
            .snaps = 0,
            .pages = p.pages,
            .pool = p.pool,
        };
        return .{
            .plan = .{ .args = args, .rows = total, .slots = windows.len },
            .tokens = base + 4 * l.tokens,
            .span = spanOf(visible, capacity),
        };
    }
};

/// A sent plan: what the launches read, the round's token ids on the device, and the keys the attention walk covers.
pub const Launch = struct { plan: pops.Plan, tokens: u64, span: usize };

/// The keys the attention walk covers when the longest row sees `visible`: min_span doubled until it holds them.
pub fn spanOf(visible: usize, capacity: usize) usize {
    var span: usize = min_span;
    while (span < visible) span *= 2;
    return @min(span, std.mem.alignForward(usize, capacity, min_span));
}

/// Mamba layer `slot`'s place in a stream's snapshots, when the round takes them.
fn snapOf(r: Rows, slot: usize) ?mamba.Snap {
    const sn = r.snaps orelse return null;
    return .{ .buf = sn.buf, .offset = r.s.states[slot].conv - r.s.mem.base(), .stride = sn.stride };
}

/// The Mamba mixer over every stream's rows: in_proj and out_proj over all of them, the recurrence a stream at a time.
fn mambaRows(
    o: Ops,
    d: model.Dims,
    act: model.Kind,
    w: model.Mamba,
    slot: usize,
    rows: []const Rows,
    x: Tensor,
    total: usize,
) forward.Error!Tensor {
    const proj = try o.project(x, w.in_proj, total, false);
    const normed: Tensor = .{ .ptr = try o.arena.take(total * d.inner() * act.size()), .kind = act };
    var at: usize = 0;
    for (rows) |r| {
        const n = r.tokens.len;
        try mamba.body(
            o,
            d,
            act,
            w,
            r.s.states[slot],
            forward.at(proj, at * d.projDim()),
            n,
            snapOf(r, slot),
            forward.at(normed, at * d.inner()),
        );
        at += n;
    }
    return o.project(normed, w.out_proj, total, w.out_proj.partial);
}

/// Every stream's rows (the plan's `tokens`, `total` of them) through every block: the final-normed rows in plan order.
pub fn forwardRound(
    ops: Ops,
    m: *const model.Model,
    place: []const usize,
    rows: []const Rows,
    l: Launch,
    trace: ?forward.Trace,
) forward.Error!Tensor {
    var o = ops;
    // a round's rows keep their decode kernels however many share it
    o.window = true;
    const d = m.dims;
    const total = l.plan.rows;
    const x = try forward.take(o, m.act, total * d.hidden);
    try o.embedRows(m.embed, l.tokens, total, x);
    for (m.blocks, place, 0..) |b, slot, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const normed = try forward.take(o, m.act, total * d.hidden);
        try o.rms(x, b.norm, normed, total, d.hidden, d.eps);
        const y = switch (b.mixer) {
            .mamba => |w| try mambaRows(o, d, m.act, w, slot, rows, normed, total),
            .attention => |w| try forward.attention(
                o,
                m.act,
                d,
                w,
                .{ .plan = .{ .p = l.plan, .slot = slot, .span = l.span } },
                normed,
                total,
            ),
            .moe => |w| try moe.layer(o, d, m.act, w, normed, total),
        };
        try hip.reduce.residual(o, m.tp, x, y, total * d.hidden);
        if (trace) |t| t.layer(t.ctx, i, x, total) catch return error.BadShape;
    }
    const out = try forward.take(o, m.act, total * d.hidden);
    try o.rms(x, m.norm_f, out, total, d.hidden, d.eps);
    return out;
}

/// The MTP head over the plan's rows (`hidden` their model rows) in pool slot `slot`: the draft rows' logits.
pub fn draftRound(
    ops: Ops,
    m: *const model.Model,
    slot: usize,
    l: Launch,
    hidden: Tensor,
) forward.Error!forward.Drafted {
    var o = ops;
    o.window = true;
    return forward.draft(
        o,
        m,
        .{ .plan = .{ .p = l.plan, .slot = slot, .span = l.span } },
        l.tokens,
        hidden,
        l.plan.rows,
    );
}

test "the span doubles from the shortest and stops at the capacity" {
    try std.testing.expectEqual(@as(usize, 128), spanOf(1, 4096));
    try std.testing.expectEqual(@as(usize, 512), spanOf(300, 4096));
    try std.testing.expectEqual(@as(usize, 4224), spanOf(5000, 4100));
}

test "the layout keeps the descriptors on even words" {
    const l = Layout.of(5, 3);
    try std.testing.expect(l.desc % 2 == 0);
    try std.testing.expectEqual(@as(usize, 3 * 5 + 3 + 3), l.count + 3);
    try std.testing.expectEqual(l.desc + 6, l.words);
}
