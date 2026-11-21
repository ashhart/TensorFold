//! DSpark's pass for the lanes (iface.Pass) over the Metal drafter: the target's taps in, blocks of drafts out.
const std = @import("std");
const host = @import("dsv41");
const dspark = @import("dspark.zig");
const target_ = @import("target.zig");

const iface = host.iface;

pub const Pass = struct {
    x: *dspark.Pass,
    /// the target whose taps the ingests name (by address)
    t: *const target_.Target,
    passes: u64 = 0,
    ingests: u64 = 0,

    pub fn pass(p: *Pass) iface.Pass {
        return .{ .ptr = p, .vtable = &.{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov } };
    }

    fn self(ptr: *anyopaque) *Pass {
        return @ptrCast(@alignCast(ptr));
    }

    fn ingest(ptr: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const p = self(ptr);
        if (rows.len == 0) return;
        const at = p.t.bufferOf(taps.address) orelse return error.UnknownTaps;
        p.ingests += 1;
        try p.x.ingest(slot, start, .{ .buf = at.buf, .off = at.off, .rows = taps.rows }, rows);
    }

    fn propose(ptr: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const p = self(ptr);
        var xs: [16]dspark.Ask = undefined;
        var ys: [16]dspark.Proposal = undefined;
        if (asks.len > xs.len or out.len != asks.len) return error.TooManySlots;
        for (asks, out, xs[0..asks.len], ys[0..asks.len]) |a, o, *x, *y| {
            x.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .params = a.params };
            y.* = .{ .drafts = o.drafts, .conf = o.conf, .cand = o.cand, .base = o.base };
        }
        p.passes += 1;
        try p.x.propose(xs[0..asks.len], ys[0..asks.len]);
    }

    fn reset(ptr: *anyopaque, slot: u32) void {
        self(ptr).x.reset(slot);
    }

    fn markov(ptr: *anyopaque) ?host.dspark.Markov {
        return self(ptr).x.markov();
    }
};

test "the pass's asks and proposals are the drafter's, field for field" {
    try std.testing.expectEqual(@sizeOf(iface.Ask), @sizeOf(dspark.Ask));
    try std.testing.expectEqual(@sizeOf(iface.Proposal), @sizeOf(dspark.Proposal));
}
