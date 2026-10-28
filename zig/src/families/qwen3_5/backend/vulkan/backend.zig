//! The Vulkan Qwen3.5-2B core behind the lane engine's backend interface: one engine state,
//! one stream at a time, GPU-greedy draws, an append-only sequence with no rewind — a new
//! prompt or a release resets the engine's buffers for real.
const std = @import("std");
const lanes = @import("lanes");
const engine_mod = @import("engine.zig");
const pinned = @import("../../config.zig");

const Engine = engine_mod.Engine;
const be = lanes.backend;

/// The advertised scheduling step: the engine cuts each prompt pass into
/// chunks of at most this many rows (per dispatch activation chunk cap).
pub const max_prompt_rows: usize = 32;

/// The SDPA arm's visible-key limit, the state's capacity ceiling (state.zig).
pub const max_capacity = @import("state.zig").max_capacity;

/// A prompt the core takes: at most `max_capacity` ids (the state ceiling;
/// the reply's decodes are admitted against it separately), each a real id.
pub fn validPrompt(ids: []const u32) bool {
    if (ids.len == 0 or ids.len > max_capacity) return false;
    for (ids) |id| if (id >= pinned.vocab) return false;
    return true;
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
            .prefill_step = prefillStepFn,
            .cancel_prefill = cancelPrefillFn,
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
        // steps floor 1: a prompt-only warm pass (max_new 0) still reads one row past the prompt
        try engine_mod.planFor(self.engine.capacity(), 0, ids.len, @max(stream.max_new, 1));
        try self.engine.reset(); // a new prompt starts from a really zeroed state (also clears a poisoned one)
        try self.engine.prefill(ids); // the same canonical 32-row schedule the stepped path runs
        stream.fill_at.store(@intCast(ids.len), .monotonic);
        self.owner = stream;
    }

    /// One 32-row chunk of the prompt pass per call; false while chunks
    /// remain. The first call validates the whole request (ids, capacity
    /// footprint with the reply's decodes) before any reset or write; later
    /// calls resume at the engine's committed visible count, so `visible()`
    /// is the progress and no cursor state exists.
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream) !bool {
        const self = cast(ptr);
        if (self.owner) |o| {
            if (o != stream) return error.QwenStreamBusy;
        } else {
            if (samplingRefused(stream)) return error.UnsupportedQwenSampling;
            const ids = stream.prompt();
            if (!validPrompt(ids)) return error.InvalidQwenPrompt;
            // steps floor 1: same footprint as the whole-pass admission (a prompt-only request reads one row past)
            try engine_mod.planFor(self.engine.capacity(), 0, ids.len, @max(stream.max_new, 1));
            try self.engine.reset(); // full request admitted: now the real zeroing
            self.owner = stream;
        }
        const ids = stream.prompt();
        const done: usize = @intCast(self.engine.visible());
        if (done >= ids.len) return true;
        const end = engine_mod.chunkEnd(done, ids.len);
        try self.engine.prefillChunk(ids[done..end]);
        const filled = self.engine.visible();
        stream.fill_at.store(@intCast(filled), .monotonic);
        return filled == ids.len;
    }

    /// A cancelled partial prompt: the engine holds a single sequence with no
    /// rewind, so keeping a prefix is impossible — a real reset releases the
    /// partially-prefilled state and the stream's ownership instead. A later
    /// release of the same stream is then a no-op.
    fn cancelPrefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        if (self.owner != stream) return error.UnknownQwenStream;
        self.owner = null;
        try self.engine.reset();
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

test "prompt validation bounds: whole requests to the ceiling, never past, real ids only" {
    try std.testing.expect(!validPrompt(&.{}));
    try std.testing.expect(validPrompt(&.{1}));
    try std.testing.expect(!validPrompt(&.{pinned.vocab})); // OOV
    var oov_last: [65]u32 = @splat(7);
    oov_last[oov_last.len - 1] = pinned.vocab; // valid length, OOV beyond the first chunk
    try std.testing.expect(!validPrompt(&oov_last));
    const over: [max_capacity + 1]u32 = @splat(7); // 7169 ids: one past the ceiling
    try std.testing.expect(!validPrompt(&over));
    const at_cap: [max_capacity]u32 = @splat(7); // 7168 ids: exactly the ceiling
    try std.testing.expect(validPrompt(&at_cap));
    const buf: [7168]u32 = @splat(7);
    for ([_]usize{ 1, 31, 32, 33, 63, 64, 65, 7105 }) |len| {
        try std.testing.expect(validPrompt(buf[0..len]));
    }
}

test "the whole-request plan gates exact fit and one past, with checked arithmetic" {
    const cap: usize = max_capacity;
    for ([_]usize{ 1, 31, 32, 33, 63, 64, 65, 7105, 7168 }) |len| {
        try engine_mod.planFor(cap, 0, len, 1); // prompt-only: last draw reads a written row
    }
    try engine_mod.planFor(cap, 0, 7167, 2); // 7167 rows + 1 decode = exact fit
    try std.testing.expectError(error.ContextTooLong, engine_mod.planFor(cap, 0, 7168, 2)); // one past
    try std.testing.expectError(error.ContextTooLong, engine_mod.planFor(cap, 0, 7169, 1));
    try std.testing.expectError(error.UnsupportedPrompt, engine_mod.planFor(cap, 0, 0, 1)); // empty
    try std.testing.expectError(error.UnsupportedPrompt, engine_mod.planFor(cap, 0, 1, 0));
    try std.testing.expectError(error.ContextTooLong, engine_mod.planFor(cap, std.math.maxInt(u64) - 2, 2, 2)); // checked add refuses, never wraps
}

test "stepped and cancel callbacks exist and their ownership guards fire before any engine access" {
    var v: Vulkan = .{ .gpa = undefined, .engine = undefined };
    const table = v.backend().vtable;
    try std.testing.expect(table.prefill_step != null); // the advertised step is wired
    try std.testing.expect(table.cancel_prefill != null);
    try std.testing.expect(table.reports_fill);
    // busy guard: a foreign stream while another owns the pass refuses before the engine is touched
    var owned: Vulkan = .{ .gpa = undefined, .engine = undefined };
    var other = try lanes.Stream.init(std.testing.allocator, .{ .id = "owned", .prompt = &.{1}, .max_new = 4 });
    defer other.deinit(std.testing.allocator);
    owned.owner = &other;
    var s = try lanes.Stream.init(std.testing.allocator, .{ .id = "t", .prompt = &.{ 1, 2 }, .max_new = 4 });
    defer s.deinit(std.testing.allocator);
    try std.testing.expectError(error.QwenStreamBusy, table.prefill_step.?(@ptrCast(&owned), &s));
    // cancel guard: no owner, or a foreign stream, refuses without touching the engine
    var fresh: Vulkan = .{ .gpa = undefined, .engine = undefined };
    try std.testing.expectError(error.UnknownQwenStream, table.cancel_prefill.?(@ptrCast(&fresh), &s));
}
