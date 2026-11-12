//! Nemotron Metal's prompt pass, whole or a chunk a call (#3021): today's cuts, the MTP head's rows in each chunk.
const std = @import("std");
const lanes = @import("lanes");
const flight = @import("prompt_flight.zig");
const Stream = lanes.Stream;

/// Streams whose prompt passes run a chunk a call (prefill_step): each one's next prompt row.
pub const Steps = struct { at: std.AutoHashMapUnmanaged(*Stream, usize) = .empty };

/// Rows of a chunk from `at` whose next token the prompt holds (`n` tokens): the MTP head absorbs these.
pub fn absorbed(at: usize, rows: usize, n: usize) usize {
    return @min(at + rows, n - 1) - at;
}

/// One chunk from `at` at today's cut (`flight.rows`), submitted on `b`; the row after it.
pub fn chunk(b: anytype, ops: anytype, s: *Stream, at: usize, next: *usize) !usize {
    const rows = flight.rows(s, at, ops.cap(), ops.decision(s), next);
    if (s.isCancelled()) return error.Cancelled; // release() waits for the chunk still in flight
    try ops.grow(s, at + rows);
    try flight.submit(b, s, ops.body(s, at, rows));
    const end = at + rows;
    s.fill_at.store(@intCast(at), .monotonic); // landed tokens only: the newest chunk is still queued on the GPU
    if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
    return end;
}

/// The whole pass (the vtable's `prefill`), as before: one chunk queued while the next is encoded.
pub fn whole(b: anytype, ops: anytype, s: *Stream) !void {
    var at = try ops.open(s);
    var next: usize = 0;
    while (at < s.prompt_len) at = try chunk(b, ops, s, at, &next);
    try ops.finish(s);
}

/// One chunk a call (prefill_step) at the whole pass's cuts; true once the prompt's last chunk is submitted.
pub fn step(b: anytype, ops: anytype, s: *Stream) !bool {
    const map = &ops.steps().at;
    const got = try map.getOrPut(b.gpa, s);
    if (!got.found_existing) got.value_ptr.* = ops.open(s) catch |e| {
        _ = map.remove(s);
        return e;
    };
    errdefer _ = map.remove(s);
    // the prompt buffer is every pass's: the chunks queued before this one land, then it holds this prompt
    try b.drain();
    ops.load(s);
    var next: usize = 0;
    const end = try chunk(b, ops, s, map.get(s).?, &next);
    if (end < s.prompt_len) {
        map.getPtr(s).?.* = end;
        return false;
    }
    _ = map.remove(s);
    try ops.finish(s);
    return true;
}

/// The host cancels between chunks: once the queue lands, the state at the pass's last chunk end is kept.
pub fn cancel(b: anytype, ops: anytype, s: *Stream) !void {
    const at = ops.steps().at.get(s) orelse return;
    try b.drain();
    s.keepInterruptedPrefill(@intCast(at));
}

test {
    _ = @import("prefill_steps_test.zig");
}
