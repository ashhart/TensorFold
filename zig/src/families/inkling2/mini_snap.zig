//! A stream's prompt-cache state: the 48 visits' K, V (full through `at`, rings' last 1,023) and conv states.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mini_model.zig");

/// A kept state: the copied rows and conv states in `buf`, in `each`'s order.
pub const Saved = struct {
    at: u32,
    visits: u32, // the plan's visit count it was made under (a state never crosses plans)
    buf: cuda.DeviceBuffer,
};

/// Positions a sliding ring keeps at `at`: a later row reads keys from its position - window + 1 on.
pub fn tail(window: usize, at: usize) usize {
    return @min(at, window - 1);
}

fn rowBytes(m: *const mm.Model) usize {
    return @as(usize, m.c.local.dim) * 2;
}

fn convWidth(m: *const mm.Model, site: mm.Site) usize {
    return if (site == .k or site == .v) @as(usize, m.c.local.kv) * m.c.local.dim else m.c.hidden;
}

/// Each copy of `slot`'s state at `at`: per visit K then V in runs of consecutive ring rows, then its conv states.
pub fn each(m: *const mm.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    const rb = rowBytes(m);
    const kvh: usize = m.c.local.kv;
    var off: usize = 0;
    for (m.w.plan.visits.slice(), 0..) |step, v| {
        const full = step.of(.full) != null;
        const len: usize = if (full) m.full.ring else m.sliding.ring;
        const from = if (full) 0 else at - tail(m.c.window, at);
        for (0..2) |side| for (0..kvh) |h| {
            const base = m.kv[side][v] + ((slot * kvh + h) * len) * rb;
            var p = from;
            while (p < at) {
                const row = p % len;
                const run = @min(at - p, len - row);
                try f(ctx, base + row * rb, off, run * rb);
                off += run * rb;
                p += run;
            }
        };
        for (std.enums.values(mm.Site)) |site| {
            const n = 3 * convWidth(m, site) * 4;
            try f(ctx, m.conv[v][@backingInt(site)] + slot * n, off, n);
            off += n;
        }
    }
}

const Count = struct {
    n: *usize,
    fn add(c: Count, _: u64, _: usize, len: usize) anyerror!void {
        c.n.* += len;
    }
};

/// Bytes a state at `at` takes.
pub fn bytes(m: *const mm.Model, at: usize) u64 {
    var n: usize = 0;
    each(m, 0, at, Count{ .n = &n }, Count.add) catch unreachable;
    return n;
}

const Copy = struct {
    m: *const mm.Model,
    s: cuda.Stream,
    buf: u64,
    out: bool, // slot -> buf (save), else buf -> slot (restore)
    fn copy(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
        const src = if (c.out) dev else c.buf + off;
        const dst = if (c.out) c.buf + off else dev;
        try c.m.d.check(c.m.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
    }
};

/// Copy `slot`'s state at `at` into new storage, queued on `s` behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, m: *const mm.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .visits = @intCast(m.w.plan.visits.len), .buf = try cuda.DeviceBuffer.alloc(m.d, @max(bytes(m, at), 256)) };
    errdefer k.buf.free();
    try each(m, slot, at, Copy{ .m = m, .s = s, .buf = k.buf.ptr, .out = true }, Copy.copy);
    return k;
}

/// Make `slot` hold `k`'s state, queued on `s` before the next chunk; a state of another plan is refused.
pub fn restore(m: *const mm.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.visits != m.w.plan.visits.len) return error.Inkling2StateOfAnotherPlan;
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, Copy{ .m = m, .s = s, .buf = k.buf.ptr, .out = false }, Copy.copy);
}

/// Free a state once the stream's queued copies of it are done.
pub fn drop(gpa: std.mem.Allocator, s: cuda.Stream, k: *Saved) void {
    s.synchronize() catch {};
    k.buf.free();
    gpa.destroy(k);
}

test "a sliding ring keeps the window less one before the next row; short prompts all of theirs" {
    try std.testing.expectEqual(@as(usize, 1023), tail(1024, 5000));
    try std.testing.expectEqual(@as(usize, 300), tail(1024, 300));
    try std.testing.expectEqual(@as(usize, 1023), tail(1024, 1023));
}
