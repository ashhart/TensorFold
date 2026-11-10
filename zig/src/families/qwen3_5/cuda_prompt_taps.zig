//! One prompt tap ring for a serialized engine; each lane retains its own verify and prompt-tail rows.
const std = @import("std");
const cuda = @import("cuda");
const ring = @import("cuda_tap_ring.zig");
pub const private_rows = 65;

pub const Stamp = struct { owner: usize, epoch: u64, range: ring.Interval };
pub const State = struct {
    buffer: ?cuda.DeviceBuffer = null,
    rows: usize = 0,
    row_bytes: usize = 0,
    held: ?Stamp = null,
    pending: ?Stamp = null,
    active: bool = false,
    prompting: bool = false,

    /// The caller has already drained the engine stream before dropping owned device storage.
    pub fn deinit(self: *State) void {
        if (self.buffer) |*b| b.free();
        self.* = .{};
    }
};

pub fn setTaps(e: anytype, taps: []const u32) !void {
    if (taps.len > e.taps.len) return error.TooManyTaps;
    for (taps) |t| if (t >= e.g.layers) return error.BadTap;
    @memcpy(e.taps[0..taps.len], taps);
    e.n_taps = taps.len;
}
pub fn tapBytes(e: anytype) usize {
    return if (e.n_taps == 0) 0 else e.tap_capacity * e.n_taps * e.g.hidden * 2;
}

const DeviceAllocation = struct {
    fn allocate(e: anytype, bytes: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(e.d, bytes);
    }
    fn zero(e: anytype, b: cuda.DeviceBuffer) !void {
        try b.fill8(0, e.stream.handle);
    }
    fn discard(e: anytype, b: *cuda.DeviceBuffer) void {
        cuda.completion.drain(e.d, e.stream, "prompt tap initialization failed before completion");
        b.free();
    }
};

/// Call only for plain dense27 DFlash2, after global multi-stream recurrent admission and before any cache.
pub fn configure(e: anytype) !void {
    return configureWith(e, DeviceAllocation);
}
pub fn configureWith(e: anytype, comptime Allocation: type) !void {
    if (e.n_taps == 0 or e.tap_capacity <= private_rows or e.prompt_taps.buffer != null) return error.BadPromptTapState;
    const pitch = try std.math.mul(usize, try std.math.mul(usize, e.n_taps, e.g.hidden), 2);
    const bytes = try std.math.mul(usize, e.tap_capacity, pitch);
    var buffer = try Allocation.allocate(e, bytes);
    errdefer Allocation.discard(e, &buffer);
    try Allocation.zero(e, buffer);
    e.prompt_taps = .{ .buffer = buffer, .rows = e.tap_capacity, .row_bytes = pitch };
    e.tap_capacity = private_rows;
}

/// Reset/reuse invalidates the old lifetime even when the same pooled Cache address comes back.
pub fn reset(c: anytype) void {
    c.tap_epoch +%= 1;
    if (c.tap_epoch == 0) c.tap_epoch = 1;
    c.tap_begin = 0;
    c.tap_end = 0;
}
pub fn invalidate(e: anytype, c: anytype) void {
    if (e.n_taps > 0 and e.prompt_taps.held != null and e.prompt_taps.held.?.owner == @intFromPtr(c))
        e.prompt_taps.held = null;
    reset(c);
}

fn privateBuffer(e: anytype, c: anytype) !cuda.DeviceBuffer {
    const b = c.tap orelse return error.NoTaps;
    const state = e.prompt_taps;
    if (e.tap_capacity != private_rows or state.rows <= private_rows or state.row_bytes == 0)
        return error.BadPromptTapState;
    const required = try std.math.mul(usize, private_rows, e.prompt_taps.row_bytes);
    const end = std.math.add(u64, b.ptr, required) catch return error.BadPromptTapBuffer;
    _ = std.math.add(u64, b.ptr, b.len) catch return error.BadPromptTapBuffer;
    const shared = e.prompt_taps.buffer orelse return error.BadPromptTapState;
    const shared_bytes = try std.math.mul(usize, state.rows, state.row_bytes);
    const shared_end = std.math.add(u64, shared.ptr, shared.len) catch return error.BadPromptTapBuffer;
    if (b.ptr == 0 or b.len < required or b.ptr % 16 != 0 or shared.ptr == 0 or shared.ptr % 16 != 0 or
        shared.len < shared_bytes or (b.ptr < shared_end and shared.ptr < end))
        return error.BadPromptTapBuffer;
    return b;
}

/// Called after admission and segment construction, before upload or any tap writer can run.
pub fn begin(e: anytype, parts: anytype, segs: anytype, prompt: bool) !void {
    if (e.n_taps == 0 or e.prompt_taps.buffer == null) return;
    const state = &e.prompt_taps;
    if (state.active or e.tap_capacity != private_rows or parts.len == 0 or parts.len != segs.len or
        (prompt and parts.len != 1)) return error.BadPromptTapState;
    for (parts, segs, 0..) |p, sg, i| {
        const dest = try privateBuffer(e, p.cache);
        if (p.cache.tap_epoch == 0 or p.rows == 0 or (!prompt and p.rows > 64) or sg.rows != p.rows or
            sg.len != p.cache.len) return error.BadPromptTapState;
        _ = try ring.retained(p.cache.len, p.rows, if (prompt) state.rows else private_rows);
        for (parts[0..i]) |prior| {
            if (prior.cache == p.cache) return error.BadPromptTapState;
            const old = try privateBuffer(e, prior.cache);
            if (old.ptr < dest.ptr + dest.len and dest.ptr < old.ptr + old.len) return error.BadPromptTapBuffer;
        }
    }
    state.active = true;
    state.prompting = prompt;
    for (parts) |p| {
        p.cache.tap_begin = 0;
        p.cache.tap_end = 0;
    }
    if (prompt) {
        const p = parts[0];
        state.held = null;
        state.pending = .{ .owner = @intFromPtr(p.cache), .epoch = p.cache.tap_epoch, .range = try ring.retained(p.cache.len, p.rows, state.rows) };
        segs[0].tap = state.buffer.?.ptr;
        segs[0].tap_rows = @intCast(state.rows);
    }
}

const DeviceCopy = struct {
    fn copy(e: anytype, dst: cuda.DeviceBuffer, dst_at: usize, src: u64, bytes: usize) !void {
        try dst.copyFrom(dst_at, src, bytes, e.stream.handle);
    }
};

/// Enqueue copies before a later prompt can overwrite the shared ring; same-stream order is the dependency.
pub fn publish(e: anytype, parts: anytype, prompt: bool) !void {
    return publishWith(e, parts, prompt, DeviceCopy);
}
pub fn publishWith(e: anytype, parts: anytype, prompt: bool, comptime Copy: type) !void {
    if (e.n_taps == 0 or e.prompt_taps.buffer == null) return;
    const state = &e.prompt_taps;
    if (!state.active or state.prompting != prompt or parts.len == 0 or (prompt and parts.len != 1))
        return error.BadPromptTapState;
    for (parts) |p| {
        const dst = try privateBuffer(e, p.cache);
        if (prompt) {
            const pending = state.pending orelse return error.BadPromptTapState;
            const full = try ring.retained(p.cache.len, p.rows, state.rows);
            if (pending.owner != @intFromPtr(p.cache) or pending.epoch != p.cache.tap_epoch or
                !std.meta.eql(pending.range, full)) return error.BadPromptTapState;
            const plan = try ring.copyTail(p.cache.len, p.rows, state.rows, private_rows);
            for (plan.items[0..plan.count]) |move| {
                try Copy.copy(e, dst, move.dest * state.row_bytes, state.buffer.?.ptr + move.source * state.row_bytes, move.rows * state.row_bytes);
            }
        }
        const held = try ring.retained(p.cache.len, p.rows, private_rows);
        p.cache.tap_begin = held.begin;
        p.cache.tap_end = held.end;
    }
    if (prompt) state.held = state.pending;
    state.pending = null;
    state.active = false;
}

/// Partial device writes must never leave apparently valid rows for a following request.
pub fn abort(e: anytype, parts: anytype) void {
    if (e.n_taps == 0 or e.prompt_taps.buffer == null) return;
    for (parts) |p| if (e.prompt_taps.held) |held| {
        if (held.owner == @intFromPtr(p.cache)) e.prompt_taps.held = null;
    };
    e.prompt_taps.pending = null;
    e.prompt_taps.active = false;
    for (parts) |p| {
        p.cache.tap_begin = 0;
        p.cache.tap_end = 0;
    }
}

/// Null preserves the original feature path; an active shared mode always returns a proven live buffer or fails.
pub fn read(e: anytype, c: anytype, start: u64, rows: usize, window: usize) !?u64 {
    if (e.n_taps == 0 or e.prompt_taps.buffer == null) return null;
    const private = try privateBuffer(e, c);
    if (c.tap_epoch == 0 or e.prompt_taps.active) return error.RowsNotHeld;
    const requested = try ring.retained(start, rows, rows);
    const own: ring.Interval = .{ .begin = c.tap_begin, .end = c.tap_end };
    if (own.contains(start, rows)) return private.ptr;
    const shared = e.prompt_taps.held orelse return error.RowsNotHeld;
    const needed = try ring.needed(start, rows, window);
    if (shared.owner != @intFromPtr(c) or shared.epoch != c.tap_epoch or requested.end > shared.range.end or
        !shared.range.contains(needed.begin, needed.end - needed.begin)) return error.RowsNotHeld;
    return e.prompt_taps.buffer.?.ptr;
}

/// Features carry absolute row positions; only the storage pointer selects the ring's modulus.
pub fn modulus(e: anytype, buffer: u64) usize {
    if (e.n_taps > 0) if (e.prompt_taps.buffer) |b| if (buffer == b.ptr) return e.prompt_taps.rows;
    return e.tap_capacity;
}

test {
    _ = @import("cuda_prompt_taps_test.zig");
}
