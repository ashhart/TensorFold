//! The Vulkan Qwen3.5-2B core behind the lane engine's backend interface: one engine state,
//! one stream at a time, GPU-greedy draws, an append-only sequence with no rewind — a new
//! prompt or a release resets the engine's buffers for real.
const std = @import("std");
const lanes = @import("lanes");
const engine_mod = @import("engine.zig");
const pinned = @import("../../config.zig");

const Engine = engine_mod.Engine;
const be = lanes.backend;

/// The core prefills a whole prompt in one pass of at most this many rows, and refuses longer ones.
pub const max_prompt_rows: usize = 32;

/// The SDPA arm's visible-key limit, the state's capacity ceiling (state.zig).
pub const max_capacity = @import("state.zig").max_capacity;

/// A prompt the core takes: at most `max_prompt_rows` ids, each a real token id.
pub fn validPrompt(ids: []const u32) bool {
    if (ids.len == 0 or ids.len > max_prompt_rows) return false;
    for (ids) |id| if (id >= pinned.vocab) return false;
    return true;
}

/// A request's writes fit the state: the prompt's rows, then a decode per further reply
/// token — the last draw reads a row already written.
pub fn fitsCapacity(prompt_len: usize, max_new: u32, capacity: usize) bool {
    return prompt_len + @as(usize, max_new -| 1) <= capacity;
}

/// A stream's request must decode greedily; the host's `greedy_only` gate refuses the rest first.
fn samplingRefused(s: *const lanes.Stream) bool {
    const sampling = s.sampling orelse return false;
    return sampling.temperature > 0;
}

pub const Vulkan = struct {
    gpa: std.mem.Allocator,
    engine: *Engine,
    owner: ?*lanes.Stream = null, // the one active stream; the engine holds a single sequence

    pub fn init(gpa: std.mem.Allocator, engine: *Engine) !*Vulkan {
        const out = try gpa.create(Vulkan);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .engine = engine };
        return out;
    }

    /// The backend's own state only: the engine (device, weights, sequence) outlives it and its
    /// owner deinits it quiescent.
    pub fn deinit(self: *Vulkan) void {
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Vulkan) lanes.Model {
        _ = self;
        return .{
            .exact_width = 1, // serial one-row exactness
            .copy_drafts = false,
            .gpu_tokens = false, // no pipelined rounds
            .mtp = false,
            .speculate = false, // no draft head
            .speculate_early = false,
            .drafts = 0,
            .hidden_rows = false, // no multi-stream shared forwards
            .streams_exact = true,
            .max_streams = 1,
            .batch_rows = 1,
            .head_trees = false,
            .draft_probabilities = false,
        };
    }

    pub fn backend(self: *Vulkan) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
            .reports_fill = true,
        } };
    }

    fn cast(ptr: *anyopaque) *Vulkan {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        if (self.owner != null) return error.QwenStreamBusy;
        if (samplingRefused(stream)) return error.UnsupportedQwenSampling;
        const ids = stream.prompt();
        if (!validPrompt(ids)) return error.InvalidQwenPrompt;
        // shortcut: prompt capped at 32 tokens, extend only after chunked recurrent-state parity is gated
        if (!fitsCapacity(ids.len, stream.max_new, self.engine.capacity())) return error.ContextTooLong;
        try self.engine.reset(); // a new prompt starts from a really zeroed state (also clears a poisoned one)
        try self.engine.prefill(ids);
        stream.fill_at.store(@intCast(ids.len), .monotonic);
        self.owner = stream;
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        if (self.owner != stream) return error.UnknownQwenStream;
        if (position != self.engine.visible()) return error.InvalidQwenPosition;
        if (samplingRefused(stream)) return error.UnsupportedQwenSampling;
        return try self.engine.draw();
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        if (self.owner != stream) return error.UnknownQwenStream;
        if (position != self.engine.visible() + 1) return error.InvalidQwenPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| std.math.cast(u32, h) orelse return error.InvalidQwenToken,
        };
        if (token >= pinned.vocab) return error.InvalidQwenToken;
        try self.engine.decode(token);
        return try self.engine.draw();
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= pinned.vocab) return error.InvalidQwenToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        if (windows.len != 1 or out.len != 1) return error.UnsupportedQwenWindow;
        const w = windows[0];
        const o = &out[0];
        if (w.parents != null or w.held != 0 or w.tokens.len != 0 or w.masks.len != 0 or w.early) return error.UnsupportedQwenWindow;
        if (w.rows() != 1) return error.UnsupportedQwenWindow;
        if (o.sampled.len != 1 or o.drafts.len != 0) return error.InvalidQwenOutputs;
        if (self.owner != w.stream) return error.UnknownQwenStream;
        if (w.positions.len != 1 or w.positions[0] != self.engine.visible() + 1) return error.InvalidQwenPosition;
        if (samplingRefused(w.stream)) return error.UnsupportedQwenSampling;
        try self.engine.decode(w.pending);
        o.sampled[0] = try self.engine.draw();
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        if (windows.len != 1 or paths.len != 1) return error.UnsupportedQwenWindow;
        const w = windows[0];
        if (self.owner != w.stream) return error.UnknownQwenStream;
        if (w.parents != null or w.held != 0) return error.UnsupportedQwenWindow;
        // The state is append-only: the one committed row is already in, and no rewind exists.
        if (paths[0].len != 1) return error.UnsupportedQwenRewind;
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.owner == stream) {
            self.owner = null;
            self.engine.reset() catch |err|
                std.log.err("qwen3.5 vulkan: the release reset failed ({s}); the next prompt resets again", .{@errorName(err)});
        }
    }
};

test "facts pin the serial one-row greedy engine" {
    const v: Vulkan = .{ .gpa = undefined, .engine = undefined };
    var cfg = try lanes.Config.init(std.testing.allocator, v.facts(), 1, 0);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), cfg.family_width);
    try std.testing.expectEqual(@as(u32, 0), cfg.max_copy);
    try std.testing.expectEqual(@as(u32, 0), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 1), cfg.batch_streams);
    try std.testing.expect(!cfg.family_streams);
    try std.testing.expect(!cfg.pipelined);
}

test "prompt validation bounds" {
    try std.testing.expect(!validPrompt(&.{}));
    try std.testing.expect(validPrompt(&.{1}));
    try std.testing.expect(!validPrompt(&.{pinned.vocab}));
    const too_long: [max_prompt_rows + 1]u32 = @splat(1);
    try std.testing.expect(!validPrompt(&too_long));
    const full: [max_prompt_rows]u32 = @splat(7);
    try std.testing.expect(validPrompt(&full));
}

test "a request's writes fit the state capacity" {
    try std.testing.expect(fitsCapacity(32, 0, 32)); // a prompt-only request writes its rows
    try std.testing.expect(fitsCapacity(4, 8, 11)); // four prompt rows and seven decodes
    try std.testing.expect(!fitsCapacity(4, 9, 11));
    try std.testing.expect(!fitsCapacity(max_prompt_rows, 2, max_prompt_rows));
}
