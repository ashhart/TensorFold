//! Readouts on the CUDA Qwen3.5 engine: a state per stream, fed each prompt piece's final-norm rows (Keep.rows).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");

/// A readout's family side (the joint schema head): its streams' states, fed each piece's rows on the engine stream.
pub const Readout = struct {
    ctx: *anyopaque,
    info: lanes.readout.Info,
    vtable: *const VTable,

    pub const State = opaque {};
    pub const VTable = struct {
        /// The stream's state for `spec` over `prompt` (its tokens give the options' lexical rows).
        open: *const fn (ctx: *anyopaque, spec: *const lanes.readout.Spec, prompt: []const u32) anyerror!*State,
        /// `count` BF16 rows of the model's width from prompt row `first_row` at `rows`, valid until the next piece.
        consume: *const fn (ctx: *anyopaque, st: *State, rows: u64, first_row: usize, count: usize, stream: cuda.Stream) anyerror!void,
        finish: *const fn (ctx: *anyopaque, st: *State, logits: []u16) anyerror!void,
        close: *const fn (ctx: *anyopaque, st: *State) void,
        /// The readout's own memory freed once its states are closed (null: the caller owns it).
        deinit: ?*const fn (ctx: *anyopaque) void = null,
    };
};

/// Rows a piece takes when the lane host names no cap, as on HIP.
pub const piece_rows = 1024;

/// The lanes' readout side: the readout (null: readout streams are refused) and each stream's progress.
pub const Side = struct {
    readout: ?Readout = null,
    open: std.AutoHashMapUnmanaged(*lanes.Stream, Open) = .empty,

    pub const Open = struct { state: *Readout.State, at: usize };

    /// Every open state closed once `e`'s queued work (a consume's) is done.
    pub fn deinit(side: *Side, gpa: std.mem.Allocator, e: *E.Engine) void {
        if (side.open.count() > 0) cuda.completion.drain(e.d, e.stream, "readout states did not drain");
        var it = side.open.valueIterator();
        while (it.next()) |o| side.readout.?.vtable.close(side.readout.?.ctx, o.state);
        side.open.deinit(gpa);
        if (side.readout) |r| if (r.vtable.deinit) |f| f(r.ctx);
    }
};

/// One piece of a readout stream's prompt pass; true once its last row is consumed (`self`: cuda_native's Lanes).
pub fn step(self: anytype, stream: *lanes.Stream, max_rows: u32) !bool {
    if (stream.isCancelled()) return error.Cancelled;
    const r = self.readout.readout orelse return error.ReadoutsUnsupported;
    if (!self.readout.open.contains(stream)) try begin(self, r, stream);
    const o = self.readout.open.getPtr(stream).?;
    const ids = stream.prompt();
    const piece: usize = @min(E.grid_rows, if (max_rows > 0) max_rows else piece_rows);
    const end = @min(ids.len, o.at + piece);
    try self.e.forward(&.{.{ .cache = try self.cacheOf(stream), .rows = end - o.at }}, ids[o.at..end], .rows);
    try r.vtable.consume(r.ctx, o.state, self.e.x, o.at, end - o.at, self.e.stream);
    o.at = end;
    stream.fill_at.store(@intCast(end), .monotonic);
    return end == ids.len;
}

/// A readout stream's first piece: a cache from row 0 (no prefix store, kept states or marks) and its state.
fn begin(self: anytype, r: Readout, stream: *lanes.Stream) !void {
    const ids = stream.prompt();
    if (ids.len == 0 or ids.len > self.capacity or self.caches.contains(stream)) return error.InvalidQwenPrompt;
    const cache = try self.take(ids.len, 0);
    errdefer self.give(cache);
    try self.caches.put(self.gpa, stream, cache);
    errdefer _ = self.caches.remove(stream);
    try self.readout.open.ensureUnusedCapacity(self.gpa, 1);
    const st = try r.vtable.open(r.ctx, stream.readout.?, ids);
    self.readout.open.putAssumeCapacity(stream, .{ .state = st, .at = 0 });
    stream.cached = 0;
}

/// A filled readout's logits from its state into the request's slice.
pub fn finish(self: anytype, stream: *lanes.Stream, logits: []u16) !void {
    const r = self.readout.readout orelse return error.ReadoutsUnsupported;
    const o = self.readout.open.get(stream) orelse return error.NotAReadout;
    if (o.at != stream.prompt().len) return error.ReadoutUnfilled;
    try r.vtable.finish(r.ctx, o.state, logits);
}

/// The stream's state closed once, whether it filled, failed or was cancelled (none before its first piece).
pub fn release(self: anytype, stream: *lanes.Stream) void {
    const kv = self.readout.open.fetchRemove(stream) orelse return;
    // a consume may still be queued on the engine stream (a cancel or a failure): it ends before its state goes
    cuda.completion.drain(self.e.d, self.e.stream, "a readout's queued work did not drain");
    self.readout.readout.?.vtable.close(self.readout.readout.?.ctx, kv.value.state);
}
