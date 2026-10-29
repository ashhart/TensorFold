//! Host callbacks used by the 27B lane backend.
const mtl = @import("metal");
const lanes = @import("tensorfold").lanes;

/// Called before a finished stream leaves its slot, its state still standing there (the reply's prompt-cache keep).
pub const Release = struct { ctx: *anyopaque, kept: *const fn (ctx: *anyopaque, s: *lanes.Stream, slot: u32) void };

/// A slot's memory on first use: `room` says whether `bytes` more fit for stream `s`, `added` takes the new buffers.
pub const Grow = struct {
    ctx: *anyopaque,
    room: *const fn (ctx: *anyopaque, bytes: usize, s: *const lanes.Stream) bool,
    added: *const fn (ctx: *anyopaque, buffers: []const mtl.Buffer) void,
};
