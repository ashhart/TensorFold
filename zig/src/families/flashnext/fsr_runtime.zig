//! Exact demand and forecast ownership at the real router boundary, sharing the existing lane engine.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const storage = @import("fsr_storage.zig");
const trace_file = @import("fsr_trace.zig");
const Commands = @import("../../core/fsr/commands.zig").Commands(mtl);
pub const Runtime = struct {
    commands: Commands = .{},
    trace: ?trace_file.Trace = null,
    trace_enabled: bool = false,
    trace_streams: usize = 1,
    fence: @import("../../core/fence.zig").Fence,
    pool: ?storage.Pool = null,
    ple: ?storage.Ple = null,
    source: storage.Source,
    keys: []usize,
    predicted: [fz.MAXR * 10]usize = undefined,
    predicted_count: usize = 0,
    layer: usize = 0,
    forecast_lg: fz.Buf,
    forecast_pick: fz.Buf,
    forecast_wts: fz.Buf,
    forecast_union: fz.Buf,
    route_rows: [fz.MAXR + 1]fz.Buf,

    pub fn init(r: *fz.Run, mtp: []const u8, ple_width: ?[2]usize, checkpoint: u64) !*Runtime {
        const s = try r.arena.create(Runtime);
        const fence = try @import("../../core/fence.zig").Fence.init(r.device);
        errdefer fence.deinit();
        var b: storage.Builder = .{ .r = r };
        const pool = if (r.window.ssd_experts) |budget| try storage.Pool.init(r, &b, mtp, budget) else null;
        const ple = if (r.window.ple_on_ssd) try storage.Ple.init(r, &b, ple_width) else null;
        var trace: ?trace_file.Trace = null;
        if (r.window.fsr_trace) |path| {
            const sink = try trace_file.Sink.open(r.arena, path);
            errdefer sink.close();
            var stride: usize = 0;
            if (pool) |p| stride = @intCast(p.plan.stride) else {
                const strides = try @import("../../core/fsr/catalog.zig").expertStrides(&b, fz.LAYERS, 512, mtp);
                for (strides) |n| stride = try std.math.add(usize, stride, n);
            }
            trace = try trace_file.Trace.init(sink, fz.LAYERS, 512, stride, checkpoint);
        }
        errdefer if (trace) |t| t.sink.close();
        s.* = .{
            .trace = trace,
            .fence = fence,
            .route_rows = undefined,
            .pool = pool,
            .ple = ple,
            .source = .{ .files = b.files.items },
            .keys = try r.arena.alloc(usize, fz.PMAX * 10),
            .forecast_lg = .{ .b = try r.buffer(fz.MAXR * 513 * 4) },
            .forecast_pick = .{ .b = try r.buffer(fz.MAXR * 10 * 4) },
            .forecast_wts = .{ .b = try r.buffer(fz.MAXR * 10 * 4) },
            .forecast_union = .{ .b = try r.buffer((1 + 2 * 320 + 320 * 32) * 4) },
        };
        for (&s.route_rows, 0..) |*buffer, i| buffer.* = try fz.i32Buf(r, &.{@intCast(i)});
        return s;
    }

    pub fn deinit(s: *Runtime) void {
        if (s.trace) |*t| {
            t.close() catch |err| std.log.warn("FSR trace incomplete: {s}", .{@errorName(err)});
            t.sink.close();
            s.trace = null;
        }
        if (s.pool) |*p| p.arena.deinit();
        if (s.ple) |*p| p.table.deinit();
        s.fence.deinit();
    }

    pub fn pause(s: *Runtime, r: *fz.Run) !void {
        try s.commands.pause(r.queue, &r.enc);
    }

    fn route(r: *fz.Run, lg: fz.Buf, rows_buffer: fz.Buf, pick: fz.Buf, wts: fz.Buf, union_: fz.Buf, rows: usize) void {
        r.enc.setPipeline(r.route_pipe);
        for ([_]fz.Buf{ lg, rows_buffer, pick, wts, union_ }, 0..) |buffer, i|
            r.enc.setBuffer(buffer.b, buffer.off, i);
        r.enc.dispatchThreads(mtl.Size.of(32 * rows, 1, 1), mtl.Size.of(32 * rows, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// The union includes every actual row and cache hit. No forecast ID can enter this demand.
    pub fn prepare(s: *Runtime, r: *fz.Run, layer: usize, lg: fz.Buf, rows_buffer: fz.Buf, pick: fz.Buf, wts: fz.Buf, rows: usize, x: fz.Buf, next_router: ?fz.Buf, guarded_rows: bool) !void {
        if (s.pool == null and (!s.trace_enabled or s.trace == null)) return;
        if (rows == 0 or rows > fz.MAXR) return error.InvalidRowCount;
        // The six-bit resident kernel encodes the upper bound, including rows ignored by a gated verdict.
        route(r, lg, s.route_rows[rows], pick, wts, r.ul, rows);
        if (s.pool != null) if (next_router) |router| {
            try r.router("q4_router_float@moe", x, router, s.route_rows[rows], s.forecast_lg, rows);
            route(r, s.forecast_lg, s.route_rows[rows], s.forecast_pick, s.forecast_wts, s.forecast_union, rows);
        };
        try s.pause(r);
        s.layer = layer;
        const encoded = @as([*]const u32, @ptrCast(@alignCast(rows_buffer.b.contents() + rows_buffer.off)))[0];
        if (encoded > rows) return error.InvalidRowCount;
        const active = if (guarded_rows) encoded else rows;
        s.predicted_count = 0;
        if (active == 0) {
            if (s.pool) |*p| {
                try p.arena.join();
                p.arena.clock.release();
            }
            return;
        }
        const ids = @as([*]const u32, @ptrCast(@alignCast(pick.b.contents() + pick.off)))[0 .. active * 10];
        for (ids, s.keys[0..ids.len]) |id, *key| {
            if (id >= 512) return error.InvalidExpert;
            key.* = layer * 512 + id;
        }
        if (s.trace_enabled) if (s.trace) |*t| try t.routeMeta(if (layer == fz.LAYERS) .mtp else if (s.trace_streams > 1) .batch else .decode, layer, ids, active, s.trace_streams);
        if (s.pool == null) return;
        try s.pool.?.arena.demand(s.keys[0..ids.len]);
        s.predicted_count = 0;
        if (next_router != null) {
            const predictions = s.forecast_pick.b.slice(u32, active * 10);
            for (predictions, s.predicted[0..predictions.len]) |id, *key| {
                if (id >= 512) return error.InvalidExpert;
                key.* = (layer + 1) * 512 + id;
            }
            s.predicted_count = predictions.len;
        }
    }

    /// Prompt routing keeps its existing full-precision router, sorting and row arithmetic.
    pub fn prompt(s: *Runtime, r: *fz.Run, layer: usize, pick: fz.Buf, rows: usize) !void {
        if (s.pool == null and (!s.trace_enabled or s.trace == null)) return;
        try s.pause(r);
        s.layer = layer;
        const ids = @as([*]const u32, @ptrCast(@alignCast(pick.b.contents() + pick.off)))[0 .. rows * 10];
        for (ids, s.keys[0..ids.len]) |id, *key| {
            if (id >= 512) return error.InvalidExpert;
            key.* = layer * 512 + id;
        }
        if (s.trace_enabled) if (s.trace) |*t| try t.routeMeta(.prompt, layer, ids, rows, 1);
        if (s.pool == null) return;
        try s.pool.?.arena.demand(s.keys[0..ids.len]);
        s.predicted_count = 0;
    }

    /// Slot publication never overlaps a writer or a GPU consumer of the same expert.
    pub fn after(s: *Runtime, r: *fz.Run) !void {
        if (s.predicted_count == 0) return;
        var missing = false;
        for (s.predicted[0..s.predicted_count]) |key| if (try s.pool.?.arena.clock.resident(key) == null) {
            missing = true;
            break;
        };
        if (!missing) {
            s.predicted_count = 0;
            return;
        }
        try s.commands.overlap(r.queue, &r.enc, s.fence);
        try s.pool.?.arena.forecast(s.predicted[0..s.predicted_count]);
        s.predicted_count = 0;
    }

    pub fn bind(s: *const Runtime, r: *fz.Run, map_at: usize, stride_at: usize, part: usize) void {
        const p = &(s.pool orelse return);
        r.enc.setBuffer(p.map.b, p.map.off + s.layer * 512 * 4, map_at);
        r.enc.setBuffer(p.strides.b, p.strides.off + part * 4, stride_at);
    }

    pub fn tables(s: *Runtime, r: *fz.Run, ids: fz.Buf, rows: usize) !?[2 + 3 * fz.GROUPS]fz.Buf {
        if (s.ple) |*p| {
            try s.pause(r);
            return try p.gather(s.source, ids, rows);
        }
        return null;
    }
};

/// Only the all-four-bit recorded gate/up variant reads GROWS; other variants encode the upper bound.
pub fn guarded(widths: anytype) bool {
    const w = widths orelse return false;
    return std.mem.eql(usize, &w.gu, &.{ 4, 32, 4, 32 });
}
