//! GLM's DFlash2 chains accepted on the GPU (glm_draft_accept.metal), as the lane core accepts them on the host.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// Threads a threadgroup: one a stream.
pub const threads = 32;

/// Rows `tokens` (pending, then drafts) against `picks`: drafts kept and the next pending token, as the host accepts.
pub fn accept(tokens: []const u32, picks: []const u32) [2]u32 {
    var kept: usize = 0;
    while (kept + 1 < tokens.len and tokens[kept + 1] == picks[kept]) kept += 1;
    return .{ @intCast(kept), picks[kept] };
}

pub const Accept = struct {
    pipe: mtl.Pipeline,

    /// `edit` replaces one text of the source (a contract control's planted fault), else null.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, edit: ?[2][]const u8) !Accept {
        const text = if (edit) |x| blk: {
            if (std.mem.indexOf(u8, sources.glm_draft_accept, x[0]) == null) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, sources.glm_draft_accept, x[0], x[1]);
        } else try a.dupe(u8, sources.glm_draft_accept);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .pipe = try mtl.Pipeline.init(device, lib, "glm_draft_accept", false) };
    }

    pub fn deinit(x: Accept) void {
        x.pipe.deinit();
    }

    /// `streams` windows' acceptance into `out` (two u32 a stream): their rows at `offsets` in `tokens` and `picks`.
    pub fn encode(x: Accept, e: mtl.ComputeEncoder, tokens: Ref, picks: Ref, offsets: Ref, rows: Ref, out: Ref, streams: u32) !void {
        if (streams == 0) return error.BadAcceptShape;
        for ([_]Ref{ tokens, picks, offsets, rows, out }) |r| if (r.off % 4 != 0) return error.BadAcceptBuffer;
        if (out.off + @as(usize, streams) * 8 > out.buf.length() or offsets.off + @as(usize, streams) * 4 > offsets.buf.length() or rows.off + @as(usize, streams) * 4 > rows.buf.length()) return error.BadAcceptBuffer;
        e.setPipeline(x.pipe);
        e.setBuffer(tokens.buf, tokens.off, 0);
        e.setBuffer(picks.buf, picks.off, 1);
        e.setBuffer(offsets.buf, offsets.off, 2);
        e.setBuffer(rows.buf, rows.off, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.setValue(streams, 5);
        e.dispatchThreads(mtl.Size.of(streams, 1, 1), mtl.Size.of(threads, 1, 1));
    }
};

test "host acceptance: drafts kept to the first differing pick, the next pending token the pick after them" {
    try std.testing.expectEqual([2]u32{ 0, 9 }, accept(&.{ 5, 7, 8 }, &.{ 9, 1, 2 })); // the first draft missed
    try std.testing.expectEqual([2]u32{ 2, 4 }, accept(&.{ 5, 7, 8 }, &.{ 7, 8, 4 })); // every draft kept
    try std.testing.expectEqual([2]u32{ 1, 3 }, accept(&.{ 5, 7, 8 }, &.{ 7, 3, 8 }));
    try std.testing.expectEqual([2]u32{ 0, 6 }, accept(&.{5}, &.{6})); // no drafts: the pick is the next token
}
