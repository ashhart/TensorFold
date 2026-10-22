//! Diagnostic class timings split command buffers, wait for each span and keep ordinary encoders when disabled.
const std = @import("std");
const mtl = @import("metal");
pub const Class = enum { dense, norm, xsum, embedding, copy, activation, gdn_prepare, gdn_scan, gdn_gate, state_publish, attention_prepare, attention, route, experts, drafter };
pub const Shape = struct { n: usize = 0, k: usize = 0, sk: usize = 0, bits: u8 = 0, calls: u64 = 0, gpu_seconds: f64 = 0 };
pub const Trace = struct {
    queue: mtl.Queue,
    enabled: bool = false,
    shapes: [32]Shape = @splat(.{}),
    shape_count: usize = 0,
    shapes_overflow: bool = false,
    calls: [@typeInfo(Class).@"enum".field_names.len]u64 = @splat(0),
    seconds: [@typeInfo(Class).@"enum".field_names.len]f64 = @splat(0),
    rounds: u64 = 0,
    report_every: u64 = 64,
    pub fn begin(t: *Trace, e: mtl.ComputeEncoder, kind: Class) Scope {
        if (!t.enabled) return .{ .encoder = e };
        const cb = t.queue.commandBuffer();
        return .{ .encoder = cb.compute(.serial), .buffer = cb, .trace = t, .kind = kind };
    }
    /// GPU seconds measured outside a Scope — a pass's own committed buffers' — join `kind`'s totals.
    pub fn record(t: *Trace, kind: Class, gpu_seconds: f64) void {
        if (!t.enabled) return;
        const index = @backingInt(kind);
        t.calls[index] += 1;
        t.seconds[index] += gpu_seconds;
    }
    /// A forward pass completed; every `report_every` of them the class totals go to the log.
    pub fn round(t: *Trace) void {
        if (!t.enabled) return;
        t.rounds += 1;
        if (t.report_every > 0 and t.rounds % t.report_every == 0) t.report();
    }
    /// The class totals, a line each, cumulative since `enabled` or the last `reset`.
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
    /// The totals and the shape table start over.
    pub fn reset(t: *Trace) void {
        t.calls = @splat(0);
        t.seconds = @splat(0);
        t.shapes = @splat(.{});
        t.shape_count = 0;
        t.shapes_overflow = false;
        t.rounds = 0;
    }
};
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

test "rounds count enabled passes only; record and reset keep the class totals" {
    var trace = Trace{ .queue = undefined };
    trace.round();
    try std.testing.expectEqual(@as(u64, 0), trace.rounds);
    trace.enabled = true;
    trace.report_every = 0; // a report goes to the log; the test does not emit one
    trace.record(.drafter, 0.5);
    trace.round();
    trace.round();
    try std.testing.expectEqual(@as(u64, 2), trace.rounds);
    try std.testing.expectEqual(@as(u64, 1), trace.calls[@backingInt(Class.drafter)]);
    try std.testing.expectEqual(@as(f64, 0.5), trace.seconds[@backingInt(Class.drafter)]);
    trace.reset();
    try std.testing.expectEqual(@as(u64, 0), trace.rounds);
    try std.testing.expectEqual(@as(u64, 0), trace.calls[@backingInt(Class.drafter)]);
}
