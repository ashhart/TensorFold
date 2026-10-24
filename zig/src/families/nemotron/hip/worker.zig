//! A tensor-parallel rank above 0: runs the steps rank 0 sends in the same order, so the collectives pair up.
const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const lanes = @import("lanes.zig");

const protocol = serve.protocol;
const Hip = lanes.Hip;

/// The rank's backend behind the shared follower: the same lanes, states and prompt cache slots, driven by the link.
pub const Worker = struct {
    pub const Lane = lanes.Lane;

    gpa: std.mem.Allocator,
    h: *Hip,

    const Follower = protocol.Follower(Worker);

    pub fn init(gpa: std.mem.Allocator, h: *Hip) Worker {
        return .{ .gpa = gpa, .h = h };
    }

    /// Runs rank 0's steps until it says stop.
    pub fn follow(w: *Worker, link: hip.collective.Link) !void {
        var f = Follower.init(w.gpa, w);
        defer f.deinit();
        try f.follow(link);
    }

    pub fn close(w: *Worker, l: *Lane) void {
        w.h.free(l);
    }

    /// prefill: id, resumed at, snapshot slot, page count, the pages, prompt length, the prompt.
    pub fn prefill(w: *Worker, f: *Follower, m: []const u32) !void {
        const table = m[5..][0..m[4]];
        const prompt = m[6 + m[4] ..][0..m[5 + m[4]]];
        f.close(m[1]);
        const l = try w.h.make();
        errdefer w.h.free(l);
        l.id = m[1];
        try l.s.set(0, table);
        if (m[3] != protocol.no_snapshot) try w.h.prefix.putSlot(m[3], l);
        l.s.len = m[2];
        try f.begin(m[1], l, prompt, m[2]);
    }

    /// The prompt's rows [from, to) through the model, as rank 0 ran them.
    pub fn fill(w: *Worker, l: *Lane, prompt: []const u32, from: usize, to: usize) !void {
        _ = try w.h.chunk(l, prompt, from, to, null);
    }

    /// verify: id, rows, the window's tokens.
    pub fn verify(w: *Worker, f: *Follower, m: []const u32) !void {
        if (m[2] > lanes.max_window) return error.WindowTooWide;
        _ = try w.h.window(try f.lane(m[1]), m[3..][0..m[2]]);
    }

    /// keep: id, rows of the last window kept.
    pub fn keep(w: *Worker, f: *Follower, m: []const u32) !void {
        try w.h.keepRows(try f.lane(m[1]), m[2]);
    }

    pub fn pages(_: *Worker, l: *Lane, first: usize, ids: []const u32) !void {
        try l.s.set(first, ids);
    }

    /// Slot `slot` takes the stream's Mamba state and last row.
    pub fn snap(w: *Worker, l: *Lane, slot: u32) !void {
        try w.h.prefix.takeSlot(slot, l);
    }

    /// Nemotron's streams write their own pages: rank 0 never copies one.
    pub fn copy(_: *Worker, _: u32, _: u32) !void {
        return error.NoPageCopies;
    }
};

test "the words of a prefill message" {
    const m = [_]u32{ @backingInt(protocol.Op.prefill), 7, 64, protocol.no_snapshot, 2, 11, 12, 3, 100, 200, 300 };
    try std.testing.expectEqualSlices(u32, &.{ 11, 12 }, m[5..][0..m[4]]);
    try std.testing.expectEqualSlices(u32, &.{ 100, 200, 300 }, m[6 + m[4] ..][0..m[5 + m[4]]]);
}
