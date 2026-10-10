//! Diagnostic class timings split command buffers, wait for each span and keep ordinary encoders when disabled.
const std = @import("std");
const mtl = @import("metal");
pub const Class = enum { dense, norm, xsum, embedding, copy, activation, gdn_prepare, gdn_scan, gdn_gate, state_publish, attention_prepare, attention };
pub const Shape = struct { n: usize = 0, k: usize = 0, sk: usize = 0, calls: u64 = 0, gpu_seconds: f64 = 0 };
pub const Trace = struct {
    queue: mtl.Queue,
    enabled: bool = false,
    shapes: [32]Shape = @splat(.{}),
    shape_count: usize = 0,
    shapes_overflow: bool = false,
    calls: [@typeInfo(Class).@"enum".field_names.len]u64 = @splat(0),
    seconds: [@typeInfo(Class).@"enum".field_names.len]f64 = @splat(0),
    pub fn begin(t: *Trace, e: mtl.ComputeEncoder, kind: Class) Scope {
        if (!t.enabled) return .{ .encoder = e };
        const cb = t.queue.commandBuffer();
        return .{ .encoder = cb.compute(.serial), .buffer = cb, .trace = t, .kind = kind };
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
            for (t.shapes[0..t.shape_count], 0..) |entry, j| if (entry.n == shape.n and entry.k == shape.k and entry.sk == shape.sk) {
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
