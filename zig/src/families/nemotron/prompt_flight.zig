//! Nemotron's existing pipelined completion boundary; a prompt observation lives until its flight lands.
const lanes = @import("lanes");

pub const Prompt = struct { stream: *lanes.Stream, rows: usize };

/// A background pass stops at its next chunk boundary; release still owns and drains the queued flight.
pub inline fn checkpoint(s: *const lanes.Stream) !void {
    if (s.isCancelled()) return error.Cancelled;
    if (s.yields()) return error.Yielded;
}

fn warming(s: *const lanes.Stream) bool {
    return s.max_new == 0 and s.yield_check != null;
}

/// Publish only landed progress: the foreground keeps one chunk queued, while a warm pass has waited for it.
pub inline fn advanced(s: *lanes.Stream, at: usize, count: usize) void {
    s.fill_at.store(@intCast(if (warming(s)) at else at - count), .monotonic);
}

/// The original prompt cuts: requested boundaries, the decision row, and the configured row cap.
pub inline fn rows(s: *const lanes.Stream, at: usize, cap: usize, decision: ?usize, next: *usize) usize {
    while (next.* < s.chunks.len and s.chunks[next.*] <= at) next.* += 1;
    var end = if (next.* < s.chunks.len) @min(s.chunks[next.*], s.prompt_len) else s.prompt_len;
    if (decision) |d| if (d >= at and d + 1 < end) {
        end = d + 1;
    };
    return @min(cap, end - at);
}

/// Foreground keeps its lookahead; a warm pass lands each chunk before checking whether another may begin.
pub inline fn submit(b: anytype, s: *lanes.Stream, body: anytype) !void {
    try b.submit(.prefill, b.next, body);
    b.flights.items[b.flights.items.len - 1].prompt = .{ .stream = s, .rows = body.rows };
    const ahead: usize = if (warming(s)) 0 else 1;
    while (b.flights.items.len > ahead) try b.land();
}

/// The real backend and recording command buffers use precisely the same wait, error and timing path.
pub inline fn land(b: anytype) !void {
    const f = b.flights.orderedRemove(0);
    defer f.cb.release();
    f.cb.wait();
    if (f.cb.failure()) |text| {
        @import("std").log.err("command buffer failed: {s}", .{text});
        return error.GpuFailed;
    }
    b.landed = @max(b.landed, f.end);
    b.last_ms = f.cb.gpuSeconds() * 1e3;
    b.gpu_ms += b.last_ms;
    b.timing.ms[@backingInt(f.kind)] += b.last_ms;
    b.timing.count[@backingInt(f.kind)] += 1;
    if (f.prompt) |p| lanes.prefill_widths.completed(b.gpa, p.stream, p.rows);
}

/// An existing failed drain may leave flights queued after release; none may keep the freed request pointer.
pub inline fn forget(b: anytype, s: *lanes.Stream) void {
    for (b.flights.items) |*f| if (f.prompt) |p| {
        if (p.stream == s) f.prompt = null;
    };
}
