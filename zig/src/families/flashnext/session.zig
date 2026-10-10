//! One admitted stream's caches and metadata beside a shared Flash Next runner and scratch plane.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const Buf = fz.Buf;

pub const Session = struct {
    m: fz.Model,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    select: [fz.CATCH]?fz.Select = @splat(null),
    shape: mtl.Buffer = undefined,
    last: Buf = undefined,
    held: [fz.MAXR]u32 = @splat(0),
    held_n: usize = 0,
    first: u32 = 0,
    used: bool = false,
    tokens: [fz.MAXR]u32 = @splat(0),
    row0: usize = 0,
    round: @import("batch_meta.zig").Round = .{},

    pub fn init(gpa: std.mem.Allocator, e: *Engine, capacity: usize, original: bool) !Session {
        var s: Session = .{ .m = e.m.* };
        errdefer s.deinit(gpa);
        s.m.cap = if (original) fz.CAP else capacity;
        s.m.pos = 0;
        s.m.state = 0;
        s.m.state_row = 0;
        s.m.recs = null;
        s.m.marks = null;
        if (!original) {
            for (&s.m.layers) |*l| if (l.linear) {
                l.cs[0] = try s.buffer(gpa, e, fz.CS_ROW);
                l.so[0] = try s.buffer(gpa, e, fz.SO_ROW);
            } else {
                try s.attention(gpa, e, l);
            };
            try s.attention(gpa, e, &s.m.mtp);
            s.m.ple.cin = try s.buffer(gpa, e, (fz.PLE_TAIL + fz.MAXR) * fz.WIDE * 2);
        }
        s.m.t.rows = try s.words(gpa, e, &.{1});
        s.m.t.mdims = try s.words(gpa, e, &.{ 1, 16, 0, 0, 0, 0, 0, 0 });
        s.m.t.pos8 = try s.buffer(gpa, e, fz.MAXR * 4);
        s.m.t.nk8 = try s.buffer(gpa, e, fz.MAXR * 4);
        s.m.t.kvmeta = try s.words(gpa, e, &.{ 0, @intCast(s.m.cap), 1 });
        s.m.t.ple_ids = try s.buffer(gpa, e, fz.MAXR * 16 * 4);
        s.shape = (try s.words(gpa, e, &.{ 1, 2, @intCast(s.m.cap), 256 })).b;
        s.last = try s.buffer(gpa, e, fz.WIDE * 2);
        for (&s.select) |*sel| {
            sel.* = try s.selector(gpa, e);
        }
        return s;
    }

    fn buffer(s: *Session, gpa: std.mem.Allocator, e: *Engine, size: usize) !Buf {
        const b = try e.r.device.buffer(@max(size, 64), fz.opts);
        errdefer b.deinit();
        try s.owned.append(gpa, b);
        @memset(b.contents()[0..@max(size, 64)], 0);
        return .{ .b = b };
    }

    fn words(s: *Session, gpa: std.mem.Allocator, e: *Engine, values: []const i32) !Buf {
        const b = try s.buffer(gpa, e, values.len * 4);
        @memcpy(b.b.slice(i32, values.len), values);
        return b;
    }

    fn attention(s: *Session, gpa: std.mem.Allocator, e: *Engine, l: anytype) !void {
        l.keys = try s.buffer(gpa, e, 2 * s.m.cap * 256 * 2);
        l.vals = try s.buffer(gpa, e, 2 * s.m.cap * 256 * 2);
        l.raw = try s.buffer(gpa, e, s.m.cap * 128 * 2);
        l.pooled = try s.buffer(gpa, e, (s.m.cap + 3) / 4 * 128 * 2);
        l.pooled_n = 0;
    }

    fn selector(s: *Session, gpa: std.mem.Allocator, e: *Engine) !?fz.Select {
        var sel = e.r.sel orelse return null;
        sel.start = try s.buffer(gpa, e, 16);
        sel.starts = try s.buffer(gpa, e, fz.CATCH * 256);
        sel.sc = try s.buffer(gpa, e, fz.MAXR * ((s.m.cap + 3) / 4) * 4);
        sel.keys = try s.buffer(gpa, e, fz.MAXR * fz.KW * 4);
        sel.complete = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.ends = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.counts = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.sparse = try s.buffer(gpa, e, fz.MAXR * 4);
        sel.pooled_shape = (try s.buffer(gpa, e, 16)).b;
        sel.q_shape = (try s.buffer(gpa, e, 16)).b;
        sel.sc_shape = (try s.buffer(gpa, e, 16)).b;
        sel.ids_shape = (try s.buffer(gpa, e, 16)).b;
        return sel;
    }

    pub fn deinit(s: *Session, gpa: std.mem.Allocator) void {
        for (s.owned.items) |b| b.deinit();
        s.owned.deinit(gpa);
    }

    pub fn normalize(s: *Session) void {
        const m = &s.m;
        if (m.state == 0 and m.state_row == 0) return;
        for (&m.layers) |*l| if (l.linear) {
            copy(l.cs[m.state].at(m.state_row * fz.CS_ROW), l.cs[0], fz.CS_ROW);
            copy(l.so[m.state].at(m.state_row * fz.SO_ROW), l.so[0], fz.SO_ROW);
        };
        m.state = 0;
        m.state_row = 0;
    }

    pub fn keep(s: *Session, n: usize) !void {
        try s.round.keep(n);
        s.m.keepRows(s.tokens[0..s.round.rows], n);
    }

    pub fn normalizeGpu(s: *Session, blit: mtl.BlitEncoder) void {
        const m = &s.m;
        if (m.state == 0 and m.state_row == 0) return;
        for (&m.layers) |*l| if (l.linear) {
            const cs = l.cs[m.state].at(m.state_row * fz.CS_ROW);
            const so = l.so[m.state].at(m.state_row * fz.SO_ROW);
            blit.copy(cs.b, cs.off, l.cs[0].b, l.cs[0].off, fz.CS_ROW);
            blit.copy(so.b, so.off, l.so[0].b, l.so[0].off, fz.SO_ROW);
        };
    }

    pub fn copy(src: Buf, dst: Buf, n: usize) void {
        std.mem.copyForwards(u8, dst.b.contents()[dst.off..][0..n], src.b.contents()[src.off..][0..n]);
    }

    pub fn bytes(capacity: usize) u64 {
        return 36 * (fz.CS_ROW + fz.SO_ROW) + 13 * (capacity * 2304 + (capacity + 3) / 4 * 256) +
            overhead(capacity);
    }

    /// Metadata and selectors are owned even when the original engine supplies this slot's caches.
    pub fn overhead(capacity: usize) u64 {
        return fz.CATCH * (fz.MAXR * ((capacity + 3) / 4) * 4 + fz.MAXR * fz.KW * 4 + 4096) + (2 << 20);
    }
};
