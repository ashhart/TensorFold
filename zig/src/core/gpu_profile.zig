//! Diagnostic class timings split command buffers, wait for each span and keep ordinary encoders when disabled.
const std = @import("std");
const mtl = @import("metal");
pub const Class = enum { dense, norm, xsum, embedding, copy, activation, gdn_prepare, gdn_scan, gdn_gate, state_publish, attention_prepare, attention, drafter };
pub const Shape = struct { n: usize = 0, k: usize = 0, sk: usize = 0, bits: u8 = 0, calls: u64 = 0, gpu_seconds: f64 = 0 };
/// The rounds between reports that TF_Q27_PROFILE=1 asks for.
pub const default_every: u64 = 64;
pub const Trace = struct {
    queue: mtl.Queue,
    enabled: bool = false,
    shapes: [32]Shape = @splat(.{}),
    shape_count: usize = 0,
    shapes_overflow: bool = false,
    calls: [@typeInfo(Class).@"enum".field_names.len]u64 = @splat(0),
    seconds: [@typeInfo(Class).@"enum".field_names.len]f64 = @splat(0),
    spanned: f64 = 0, // the spans' own command buffers' GPU seconds (a `record` is not a span)
    rounds: u64 = 0, // forward passes run while enabled
    report_every: u64 = default_every, // the class totals go to the log every this many rounds (0: only at close)
    holds: u32 = 0, // open draft frames: a span keeps the frame's encoder, so the frame's pending ops stay in order
    pub fn begin(t: *Trace, e: mtl.ComputeEncoder, kind: Class) Scope {
        if (!t.enabled or t.holds != 0) return .{ .encoder = e };
        const cb = t.queue.commandBuffer();
        return .{ .encoder = cb.compute(.serial), .buffer = cb, .trace = t, .kind = kind };
    }
    /// GPU seconds timed outside a Scope (a draft frame's own buffers) join `kind`'s totals.
    pub fn record(t: *Trace, kind: Class, gpu_seconds: f64) void {
        if (!t.enabled) return;
        t.calls[@backingInt(kind)] += 1;
        t.seconds[@backingInt(kind)] += gpu_seconds;
    }
    /// A forward pass completed; every `report_every` of them the class totals go to the log.
    pub fn round(t: *Trace) void {
        if (!t.enabled) return;
        t.rounds += 1;
        if (t.report_every > 0 and t.rounds % t.report_every == 0) t.report();
    }
    /// The class totals since the trace began, a log line each.
    pub fn report(t: *const Trace) void {
        if (t.rounds == 0) return;
        var total: f64 = 0;
        for (t.seconds) |seconds| total += seconds;
        std.log.info("gpu profile: {d} rounds, {d:.1} ms", .{ t.rounds, total * 1e3 });
        for (std.enums.values(Class), 0..) |kind, i| {
            if (t.calls[i] == 0) continue;
            std.log.info("gpu profile: {s:<18} {d:>6} calls {d:>9.2} ms {d:>8.3} ms/round", .{ @tagName(kind), t.calls[i], t.seconds[i] * 1e3, t.seconds[i] * 1e3 / @as(f64, @floatFromInt(t.rounds)) });
        }
    }
    /// At close: the totals, unless the last round just logged them.
    pub fn reportRest(t: *const Trace) void {
        if (t.report_every == 0 or t.rounds % t.report_every != 0) t.report();
    }
};
/// A draft frame's hold on its trace from begin to release: the trace it took is the one it gives back.
pub const Hold = struct {
    trace: ?*Trace = null,
    pub fn take(h: *Hold, trace: ?*Trace) void {
        std.debug.assert(h.trace == null);
        const t = trace orelse return;
        t.holds += 1;
        h.trace = t;
    }
    pub fn release(h: *Hold) void {
        const t = h.trace orelse return;
        std.debug.assert(t.holds > 0);
        t.holds -= 1;
        h.trace = null;
    }
};
/// A stretch outside the profile (a served prompt pass): spans stay unsplit and nothing counts until `end`.
pub const Pause = struct {
    trace: ?*Trace = null,
    pub fn end(p: Pause) void {
        if (p.trace) |t| t.enabled = true;
    }
};
pub fn pause(trace: ?*Trace) Pause {
    const t = trace orelse return .{};
    if (!t.enabled) return .{};
    t.enabled = false;
    return .{ .trace = t };
}
/// TF_Q27_PROFILE's rounds between reports: unset, empty or 0 is off, 1 is every 64, N is every N; else invalid.
pub fn every(value: ?[]const u8) error{InvalidProfile}!?u64 {
    const v = value orelse return null;
    if (v.len == 0) return null;
    for (v) |c| if (!std.ascii.isDigit(c)) return error.InvalidProfile;
    const n = std.fmt.parseInt(u64, v, 10) catch return error.InvalidProfile;
    return switch (n) {
        0 => null,
        1 => default_every,
        else => n,
    };
}
pub const Scope = struct {
    encoder: mtl.ComputeEncoder,
    buffer: ?mtl.CommandBuffer = null,
    trace: ?*Trace = null,
    kind: Class = .dense,
    closed: bool = false,
    shape: ?Shape = null,
    pub fn cancel(s: *Scope) void {
        if (s.buffer != null and !s.closed) {
            s.encoder.end();
            s.closed = true;
        }
    }
    pub fn end(s: *Scope) !void {
        const cb = s.buffer orelse return;
        s.encoder.end();
        s.closed = true;
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.ProfileGpuFailure;
        const t = s.trace.?;
        const index = @backingInt(s.kind);
        t.calls[index] += 1;
        t.seconds[index] += cb.gpuSeconds();
        t.spanned += cb.gpuSeconds();
        if (s.shape) |shape| {
            var found: ?usize = null;
            for (t.shapes[0..t.shape_count], 0..) |entry, j| if (entry.n == shape.n and entry.k == shape.k and entry.sk == shape.sk and entry.bits == shape.bits) {
                found = j;
                break;
            };
            if (found == null and t.shape_count < t.shapes.len) {
                found = t.shape_count;
                t.shapes[t.shape_count] = shape;
                t.shape_count += 1;
            }
            if (found) |j| {
                t.shapes[j].calls += 1;
                t.shapes[j].gpu_seconds += cb.gpuSeconds();
            } else t.shapes_overflow = true;
        }
    }
};
pub fn begin(trace: ?*Trace, e: mtl.ComputeEncoder, kind: Class) Scope {
    return if (trace) |t| t.begin(e, kind) else .{ .encoder = e };
}

test "disabled profiling keeps the caller encoder and records no spans" {
    var trace = Trace{ .queue = undefined };
    var span = trace.begin(undefined, .dense);
    try span.end();
    span.cancel();
    try std.testing.expect(span.buffer == null);
    try std.testing.expectEqual(@as(u64, 0), trace.calls[0]);
}

test "gpu profile: rounds and records count only while enabled; a record is not a span" {
    var trace = Trace{ .queue = undefined, .report_every = 0 }; // 0: the test writes no log lines
    trace.round();
    trace.record(.drafter, 0.5);
    try std.testing.expectEqual(@as(u64, 0), trace.rounds);
    try std.testing.expectEqual(@as(u64, 0), trace.calls[@backingInt(Class.drafter)]);
    trace.enabled = true;
    trace.record(.drafter, 0.5);
    trace.round();
    trace.round();
    try std.testing.expectEqual(@as(u64, 2), trace.rounds);
    try std.testing.expectEqual(@as(u64, 1), trace.calls[@backingInt(Class.drafter)]);
    try std.testing.expectEqual(@as(f64, 0.5), trace.seconds[@backingInt(Class.drafter)]);
    try std.testing.expectEqual(@as(f64, 0), trace.spanned);
}

test "gpu profile: a held trace keeps the frame's encoder, and each hold gives back the trace it took" {
    var trace = Trace{ .queue = undefined, .enabled = true };
    var first: Hold = .{};
    var second: Hold = .{};
    first.take(&trace);
    second.take(&trace);
    var span = trace.begin(undefined, .dense);
    try span.end();
    try std.testing.expect(span.buffer == null);
    try std.testing.expectEqual(@as(u64, 0), trace.calls[@backingInt(Class.dense)]);
    first.release();
    try std.testing.expectEqual(@as(u32, 1), trace.holds);
    second.release();
    second.release(); // a hold gives its trace back once; no count wraps
    try std.testing.expectEqual(@as(u32, 0), trace.holds);
    first.take(null); // no profiler: no hold
    first.release();
    try std.testing.expect(first.trace == null);
}

test "gpu profile: a paused trace splits and counts nothing, then runs as before" {
    var trace = Trace{ .queue = undefined, .enabled = true, .report_every = 0 };
    const paused = pause(&trace);
    var span = trace.begin(undefined, .attention);
    try span.end();
    trace.round();
    trace.record(.drafter, 1);
    try std.testing.expect(span.buffer == null);
    paused.end();
    try std.testing.expect(trace.enabled);
    try std.testing.expectEqual(@as(u64, 0), trace.rounds);
    try std.testing.expectEqual(@as(u64, 0), trace.calls[@backingInt(Class.drafter)]);
    var off = Trace{ .queue = undefined };
    pause(&off).end(); // a disabled trace stays disabled
    try std.testing.expect(!off.enabled);
    pause(null).end();
}

test "gpu profile: TF_Q27_PROFILE is off unless 1 (every 64 rounds) or a round count" {
    try std.testing.expectEqual(@as(?u64, null), try every(null));
    try std.testing.expectEqual(@as(?u64, null), try every(""));
    try std.testing.expectEqual(@as(?u64, null), try every("0"));
    try std.testing.expectEqual(@as(?u64, null), try every("000"));
    try std.testing.expectEqual(@as(?u64, default_every), try every("1"));
    try std.testing.expectEqual(@as(?u64, 16), try every("16"));
    for ([_][]const u8{ "on", "true", "-1", "+2", " 1", "1 ", "1_0", "0x10", "1.5", "99999999999999999999" }) |v| try std.testing.expectError(error.InvalidProfile, every(v));
}
