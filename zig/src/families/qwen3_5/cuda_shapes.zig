//! Captured launch geometry, separate from the live counts read through the staged device tables.
const std = @import("std");
const cuda = @import("cuda");
const Round = @import("cuda_round.zig");

pub const Shape = struct {
    total: u16, // a prompt chunk's rows (up to Round.max_rows, never captured), or a window bucket
    nseg: u8,
    record: bool,
    prompt: bool,
    taps: bool,
    slots: u8,
    hidden: bool = false,
    paged: bool = false,
    /// A pipeline stage's piece of the forward (cuda_stage_forward.zig); one engine runs the whole.
    part: @import("cuda_stage_forward.zig").Part = .whole,

    pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
        std.debug.assert(rows > 0 and rows <= @as(usize, if (record) Round.most_verify else Round.max_rows) and segments > 0 and segments <= rows);
        const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
        const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
        return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
    }

    pub fn key(s: Shape) u32 {
        // captured shapes only: a window bucket's rows fit the low byte
        return @as(u32, @as(u8, @intCast(s.total))) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, s.slots) << 19 | @as(u32, @intFromBool(s.hidden)) << 27 | @as(u32, @intFromBool(s.paged)) << 28 | @as(u32, @backingInt(s.part)) << 29;
    }
};

/// One shape's launches (the engine's `launchShape`): uncaptured for a prompt chunk or the exact path, else replayed.
pub fn launch(e: anytype, shape: Shape) !void {
    // a prompt chunk's GEMMs (a window's rows keep one-row steps' bits) dwarf its launches: never captured
    if (e.learning != null or e.pockets.any() or shape.prompt) return e.body(shape);
    const gop = try e.graphs.getOrPut(e.gpa, shape.key());
    if (!gop.found_existing) {
        errdefer _ = e.graphs.remove(shape.key());
        // captured, nothing runs: the replay below is this shape's first run
        try cuda.graph.beginCapture(e.stream, .thread_local);
        e.body(shape) catch |err| {
            var broken = cuda.graph.endCapture(e.stream) catch return err;
            broken.deinit();
            return err;
        };
        var captured = try cuda.graph.endCapture(e.stream);
        defer captured.deinit();
        gop.value_ptr.* = try captured.instantiate();
    }
    try gop.value_ptr.launchOn(e.stream);
}
