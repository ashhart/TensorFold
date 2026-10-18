//! Bounded investigation snapshots; file I/O and independent CPU work are not performance measurements.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const Reference = @import("reference.zig").Reference;
const Stage = tf.deepseek_v41_profiler.Stage;
const widths = [_]usize{ 20480 * 2, 5120 * 2, 5120 * 2, 6 * 4, 6 * 4 };
const stride = 61488;
const bytes = stride * 80;

const Capture = struct {
    buffer: mtl.Buffer,
    pipeline: mtl.Pipeline,
    cpu: []u8,

    fn gpu(ctx: *anyopaque, enc: mtl.ComputeEncoder, layer: u32, stage: Stage, model: *const Model) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        var destination: usize = (@as(usize, layer) * 2 + @intFromBool(stage == .moe)) * stride;
        const ws = &model.workspace;
        inline for (.{ ws.view(.stream), ws.view(.normalized), ws.view(.branch), ws.view(.route_ids), ws.view(.route_weights) }, widths, 0..) |source, count, field| {
            if (stage == .moe or field < 3) {
                enc.setPipeline(self.pipeline);
                enc.setBuffer(source.buffer, source.offset, 0);
                enc.setBuffer(self.buffer, destination, 1);
                enc.setValue(@as(u32, @intCast(count / 4)), 2);
                enc.dispatchThreads(.{ .width = count / 4 }, .{ .width = 256 });
                destination += count;
            }
        }
        enc.barrier();
    }
    fn reference(ctx: *anyopaque, layer: usize, stage: Stage, ref: *const Reference) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        const start = (layer * 2 + @intFromBool(stage == .moe)) * stride;
        var destination: []u8 = self.cpu[start..][0..stride];
        inline for (.{ std.mem.sliceAsBytes(&ref.h), std.mem.sliceAsBytes(&ref.input), std.mem.sliceAsBytes(&ref.output), std.mem.sliceAsBytes(&ref.last_routes), std.mem.sliceAsBytes(&ref.last_routing) }, 0..) |source, field| {
            if (stage == .moe or field < 3) {
                @memcpy(destination[0..source.len], source);
                destination = destination[source.len..];
            }
        }
    }
};
fn normalizedError(a: []const u8, b: []const u8) !f64 {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 2) {
        const x: f32 = @bitCast(@as(u32, std.mem.readInt(u16, a[i..][0..2], .little)) << 16);
        const y: f32 = @bitCast(@as(u32, std.mem.readInt(u16, b[i..][0..2], .little)) << 16);
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteTraceValues;
        squared += (@as(f64, x) - y) * (@as(f64, x) - y);
        denominator += @as(f64, y) * y;
    }
    return if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator);
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !void {
    if (ids.len == 0 or ids.len > 16) return error.InvalidOracleWindow;
    const lib = try mtl.Library.fromSource(model.device, "#include <metal_stdlib>\nusing namespace metal;\nkernel void trace_copy(const device uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<n) y[i]=x[i]; }", .{});
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(model.device, lib, "trace_copy", false);
    defer pipeline.deinit();
    const buffer = try model.device.buffer(bytes, mtl.ResourceOptions.shared);
    defer buffer.deinit();
    const cpu = try gpa.alloc(u8, bytes);
    defer gpa.free(cpu);
    var capture = Capture{ .buffer = buffer, .pipeline = pipeline, .cpu = cpu };
    model.residency.add(buffer);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        model.residency.remove(buffer);
        model.residency.commit();
    }
    var reference = try Reference.init(gpa, model.cfg.text_config, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer reference.deinit();
    reference.observer = .{ .ctx = &capture, .read = Capture.reference };
    const expected = try gpa.alloc(f32, ids.len * 129280);
    defer gpa.free(expected);
    try model.reset();
    for (ids, 0..) |id, i| @memcpy(expected[i * 129280 ..][0..129280], try model.forward(&.{id}, .last));
    try model.reset();
    const previous_observer = model.observer;
    model.observer = .{ .ctx = &capture, .encode = Capture.gpu };
    defer model.observer = previous_observer;
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "[\n");
    for (ids, 0..) |id, token| {
        @memset(cpu, 0);
        @memset(buffer.contents()[0..bytes], 0);
        const actual = try model.forward(&.{id}, .last);
        for (actual, expected[token * 129280 ..][0..129280]) |a, b|
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return error.TraceChangedLogits;
        _ = try reference.step(id);
        const gpu = buffer.contents()[0..bytes];
        for (0..80) |point| {
            const offset = point * stride;
            const g = gpu[offset..][0..stride];
            const c = cpu[offset..][0..stride];
            const stream_error = try normalizedError(g[0..40960], c[0..40960]);
            const input_error = try normalizedError(g[40960..51200], c[40960..51200]);
            const branch_error = try normalizedError(g[51200..61440], c[51200..61440]);
            const gr: [*]const u32 = @ptrCast(@alignCast(g.ptr + 61440));
            const cr: [*]const u32 = @ptrCast(@alignCast(c.ptr + 61440));
            const routes_match = point % 2 == 0 or std.mem.eql(u32, gr[0..6], cr[0..6]);
            try report.print(gpa, "{s}{{\"token\":{d},\"layer\":{d},\"stage\":\"{s}\",\"stream_nrms\":{d},\"input_nrms\":{d},\"branch_nrms\":{d},\"routes_match\":{s}}}", .{ if (token == 0 and point == 0) "" else ",\n", token, point / 2, if (point % 2 == 0) "attention" else "moe", stream_error, input_error, branch_error, if (routes_match) "true" else "false" });
            if (branch_error > 0.01 or !routes_match) std.debug.print("trace token={d} layer={d} stage={s} stream={d} input={d} branch={d} routes_match={}\n", .{ token, point / 2, if (point % 2 == 0) "attention" else "moe", stream_error, input_error, branch_error, routes_match });
        }
        inline for (.{ "gpu", "cpu" }, .{ gpu, cpu }) |name, data| {
            const path = try std.fmt.allocPrint(gpa, "{s}/trace-{d}-{s}.bin", .{ dir, token, name });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        }
        std.debug.print("trace captured token {d}/{d}, same_logits=true\n", .{ token + 1, ids.len });
    }
    try report.appendSlice(gpa, "\n]\n");
    const path = try std.fs.path.join(gpa, &.{ dir, "layer-trace.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
}
