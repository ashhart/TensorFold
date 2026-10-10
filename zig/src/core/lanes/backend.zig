//! What the round loop asks a backend (Metal, CUDA or the fake) to compute; drawn tokens stay on the GPU behind handles.
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Shape = @import("shape.zig").Shape;
const Held = @import("stream.zig").Held;
const Row = @import("logprob.zig").Row;

/// A token to feed: one the backend drew and holds (a handle), or a host value (forced or pending).
pub const Feed = union(enum) { handle: u64, value: u32 };

/// One stream's verify window: its pending row, then its drafts, each row drawn at its own keyed position.
pub const Window = struct {
    stream: *Stream,
    pending: u32,
    held: u32, // rows after the pending one taken from the drafts the backend holds for the stream
    tokens: []const u32, // drafts known on the host (forced, copied, tree); empty when held > 0
    parents: ?[]const i32, // each row's parent row (row 0: -1) for a tree; null for a chain
    positions: []const u64, // the absolute position each row's draw is keyed at
    early: bool = false, // also draft each row's first head draft behind the verify (speculate_early)
    masks: []const u32 = &.{}, // structured output: each row's allowed-token bits (rows x words, lanes/grammar.zig); empty: none

    pub fn rows(w: Window) usize {
        return 1 + @as(usize, w.held) + w.tokens.len;
    }
};

/// A drafted level's best tokens by the head (the first is the held draft) and the head's probability for each.
pub const Alternative = struct { tokens: [4]u32, probs: [4]f64 };

/// A window's results: each row's drawn token, the held drafts the forward verified, and logprob rows when asked.
pub const Verified = struct { sampled: []u32, drafts: []u32, rows: []Row = &.{} };

/// Tapped cache rows for a drafter: the prompt's until the first verify, then each verify's kept rows until the next.
pub const Features = struct {
    pub const Space = enum { host, device };
    pub const Dtype = enum { bf16, f16, f32, u32 };
    pub const Fence = union(enum) { none, cuda_event: u64, metal_event: u64 };
    buffer: u64, // a host pointer, or a device pointer on `device`
    offset: u64 = 0, // bytes into `buffer` where the first row starts
    rows: u32, // packed rows, row-major
    row_bytes: u32, // a row is the drafter's taps in its `taps` order, each the hidden width of `dtype`
    space: Space,
    device: i32 = 0, // the drafter must run on this device
    dtype: Dtype,
    ready: Fence = .none, // .none: written; else the fence to wait on before reading, kind naming the backend
};

/// Absorb a stream's kept rows into its draft head and hold `depth` drafts for its next round.
pub const DraftRequest = struct {
    stream: *Stream,
    follow: []const u32, // the token after each kept row (the last is the stream's pending token)
    first: ?Feed = null, // the prompt's first token when it is still on the device (follow is then empty)
    rows: ?[]const u32, // the kept rows of the last verify; null: the prompt's last row
    start: u64, // the cache length the last verify started from
    position: u64, // where the first draft lands (the new cache length + 1)
    depth: u32, // drafts to hold (0: absorb only)
    early: bool = false, // settle the drafts the verify began (speculate_early) instead of drafting late
    lanes: ?*const Shape = null, // a tree of drafts to hold, each lane its parent's rank-r head token (depth unused)
    ranks: bool = false, // also keep each chained level's best tokens (alternatives(), for grafts)
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Prefill the stream's prompt into its caches (and its draft head's).
        prefill: *const fn (ptr: *anyopaque, s: *Stream) anyerror!void,
        /// One chunk of the prompt pass, false while chunks remain; null: the pass runs whole.
        prefill_step: ?*const fn (ptr: *anyopaque, s: *Stream) anyerror!bool = null,
        /// Draw the first token from the prompt's last row at `position`; a handle to it.
        first: *const fn (ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64,
        /// Feed one token and queue the draw of the next at `position` (a one-token round); a handle to it.
        queue: *const fn (ptr: *anyopaque, s: *Stream, feed: Feed, position: u64) anyerror!u64,
        /// The value of a drawn token (waits for the GPU).
        read: *const fn (ptr: *anyopaque, handle: u64) anyerror!u32,
        /// One forward over every window, each row drawn with its stream's sampling at its position.
        verify: *const fn (ptr: *anyopaque, windows: []const Window, out: []Verified) anyerror!void,
        /// Roll each stream's caches back to its kept rows (a prefix count, or a tree's path).
        keep: *const fn (ptr: *anyopaque, windows: []const Window, paths: []const []const u32) anyerror!void,
        /// Absorb kept rows and hold drafts for the next round (one batch for a shared round).
        draft: *const fn (ptr: *anyopaque, requests: []const DraftRequest) anyerror!void,
        /// Undo an early speculation whose round was cut (the drafts come from a late request instead).
        unspeculate: ?*const fn (ptr: *anyopaque, s: *Stream) anyerror!void = null,
        /// Each held draft's chance of landing, for heads that give them; false when it has none.
        probabilities: ?*const fn (ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool = null,
        /// A tree head's held drafts as host tokens and parents (gpa-owned), read where Python reads them.
        tree: ?*const fn (ptr: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?Held = null,
        /// The head's best tokens and probabilities at each level it drafted for the stream (waits for the drafts).
        alternatives: ?*const fn (ptr: *anyopaque, s: *Stream, out: []Alternative) anyerror!usize = null,
        /// The drafter's layers, told once before any forward so the target keeps only those (null: built with them).
        prepare_features: ?*const fn (ptr: *anyopaque, taps: []const u32) anyerror!void = null,
        /// Cache rows [start, start + count) of the prepared taps, within what `Features` says is held (null: none).
        features: ?*const fn (ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!Features = null,
        /// The first token's logprob row (the prompt's last row) for a stream with `logprobs`; null: not given.
        first_row: ?*const fn (ptr: *anyopaque, s: *Stream) anyerror!Row = null,
        /// `first` with the first token drawn among `mask`'s allowed tokens (structured output; Info.structures backends).
        first_masked: ?*const fn (ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 = null,
        /// The stream left the rounds: free its caches and held drafts.
        release: *const fn (ptr: *anyopaque, s: *Stream) void,
    };

    pub fn prefill(b: Backend, s: *Stream) !void {
        return b.vtable.prefill(b.ptr, s);
    }
    pub fn first(b: Backend, s: *Stream, position: u64) !u64 {
        return b.vtable.first(b.ptr, s, position);
    }
    pub fn queue(b: Backend, s: *Stream, feed: Feed, position: u64) !u64 {
        return b.vtable.queue(b.ptr, s, feed, position);
    }
    pub fn read(b: Backend, handle: u64) !u32 {
        return b.vtable.read(b.ptr, handle);
    }
    pub fn verify(b: Backend, windows: []const Window, out: []Verified) !void {
        return b.vtable.verify(b.ptr, windows, out);
    }
    pub fn keep(b: Backend, windows: []const Window, paths: []const []const u32) !void {
        return b.vtable.keep(b.ptr, windows, paths);
    }
    pub fn draft(b: Backend, requests: []const DraftRequest) !void {
        return b.vtable.draft(b.ptr, requests);
    }
    pub fn features(b: Backend, s: *Stream, taps: []const u32, start: u64, count: u32) !Features {
        const f = b.vtable.features orelse return error.NoFeatures;
        return f(b.ptr, s, taps, start, count);
    }
    pub fn firstRow(b: Backend, s: *Stream) !Row {
        const f = b.vtable.first_row orelse return error.LogprobsUnsupported;
        return f(b.ptr, s);
    }
    pub fn release(b: Backend, s: *Stream) void {
        b.vtable.release(b.ptr, s);
    }

    /// A driver of its own starts a stream: the prompt pass and its first token, committed; a cancel or failure releases it.
    pub fn opening(b: Backend, gpa: std.mem.Allocator, s: *Stream) !u32 {
        errdefer b.release(s);
        try b.prefill(s);
        const token = try b.read(try b.first(s, s.prompt_len));
        _ = try s.commit(gpa, &.{token}, &.{});
        return token;
    }
};

/// Where the round loop reads time: the depth rule's costs and a shared round's overhead.
pub const Mark = enum { cost, overhead };

pub const Clock = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        start: *const fn (ptr: *anyopaque) void,
        elapsed_ms: *const fn (ptr: *anyopaque, mark: Mark) f64,
    };

    pub fn start(c: Clock) void {
        c.vtable.start(c.ptr);
    }
    pub fn elapsedMs(c: Clock, mark: Mark) f64 {
        return c.vtable.elapsed_ms(c.ptr, mark);
    }
};

/// The monotonic clock production rounds read (Python time.perf_counter: CLOCK_UPTIME_RAW on macOS).
pub const WallClock = struct {
    io: std.Io,
    started: std.Io.Timestamp = .zero,

    pub fn clock(w: *WallClock) Clock {
        return .{ .ptr = w, .vtable = &.{ .start = startFn, .elapsed_ms = elapsedFn } };
    }

    fn startFn(ptr: *anyopaque) void {
        const w: *WallClock = @ptrCast(@alignCast(ptr));
        w.started = std.Io.Timestamp.now(w.io, .awake);
    }

    fn elapsedFn(ptr: *anyopaque, _: Mark) f64 {
        const w: *WallClock = @ptrCast(@alignCast(ptr));
        const d = w.started.durationTo(std.Io.Timestamp.now(w.io, .awake));
        return @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
    }
};
