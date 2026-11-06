//! Native model resources with stable addresses and live shared-device admission.
const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const c = @import("config.zig");
const Weights = @import("hip_weights.zig").Weights;
const Ops = @import("hip_ops.zig").Ops;
const Graph = @import("hip_text.zig").Graph;

pub const rows = 32;
const reserve = 512 << 20;
const image_reserve = 2 << 30;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    blas: hip.blas.Blas,
    flash: hip.attention_aotriton.Library,
    ops: Ops,
    weights: Weights,
    graph: Graph,
    launcher: hip.Launcher,
    arena: hip.Arena,
    drawer: serve.draw.Drawer,
    input: hip.DeviceBuffer,
    window: usize,
    streams: usize,
    cache_bytes: usize,
    cache_slots: usize,

    pub fn load(a: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: serve.Options) !*Engine {
        if (o.window == 0 or o.window > c.context or o.streams == 0 or o.world != 1 or o.rank != 0) return error.Invalid;
        var layout = try @import("hip_weights.zig").Layout.init(a);
        defer layout.deinit();
        const graph_bytes = try Graph.memoryBytes(@max(o.window, rows));
        const lane_bytes = o.window * c.layers * c.kv * 4;
        const fixed = layout.bytes + graph_bytes + o.window * (c.hidden * 2 + 12) + reserve + (if (o.vision != null) @as(usize, image_reserve) else 0) + (2 << 20);
        const e = try a.create(Engine);
        errdefer a.destroy(e);
        e.gpa = a;
        e.io = io;
        e.window = o.window;
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        _ = try hip.Device.kernelCaps(&e.driver, dev.index);
        e.ctx = try hip.Context.init(&e.driver, dev.index);
        errdefer e.ctx.deinit();
        const pool = try hip.admission.budget(a, io, e.ctx);
        const room = pool.room(hip.usage(false).device);
        if (fixed >= room or room - fixed < lane_bytes) return error.GlmOcrModelDoesNotFit;
        e.streams = @min(o.streams, @min(4, (room - fixed) / lane_bytes));
        if (o.fixed and e.streams != o.streams) return error.GlmOcrParallelDoesNotFit;
        const cache_room = (if (o.cache_over_cap) pool.free -| pool.reserve else room) -| fixed -| (e.streams * lane_bytes);
        e.cache_bytes = try cacheBudget(o.cache_gib, @intCast(cache_room), dev.policy.slots);
        e.cache_slots = dev.policy.slots;
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        e.blas = try hip.blas.Blas.open(e.stream);
        errdefer e.blas.close();
        e.flash = try hip.attention_aotriton.Library.open(e.stream);
        errdefer e.flash.close(e.stream);
        e.ops = try Ops.loadOn(&e.driver, dev.index);
        errdefer e.ops.unload();
        e.weights = try Weights.load(a, &e.driver, io, dir, layout.bytes);
        errdefer e.weights.deinit();
        e.graph = try Graph.init(e.stream, &e.weights, &e.ops, &e.blas, &e.flash, @max(o.window, rows), graph_bytes);
        errdefer e.graph.deinit();
        e.launcher = try hip.Launcher.load(&e.driver, dev.policy.choices(), hip.kernels.images);
        errdefer e.launcher.unload();
        e.arena = try hip.Arena.init(&e.driver, 1 << 20);
        errdefer e.arena.deinit();
        e.drawer = try serve.draw.Drawer.init(a, &e.driver, .bf16, c.vocab, rows);
        errdefer e.drawer.deinit();
        e.input = try hip.DeviceBuffer.alloc(&e.driver, rows * (c.hidden * 2 + 12));
        return e;
    }
    pub fn available(e: *const Engine) !usize {
        return (try e.ctx.memInfo()).free -| reserve;
    }
    pub fn drawOps(e: *Engine) hip.ops.Ops {
        return .{ .l = &e.launcher, .bf16 = true, .stream = e.stream.handle, .arena = &e.arena };
    }
    pub fn deinit(e: *Engine) void {
        e.ctx.makeCurrent() catch {};
        e.stream.synchronize() catch {};
        e.input.free();
        e.drawer.deinit();
        e.arena.deinit();
        e.launcher.unload();
        e.graph.deinit();
        e.weights.deinit();
        e.ops.unload();
        e.flash.close(e.stream);
        e.blas.close();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }
};

pub fn cacheBudget(gib: ?f64, room: usize, slots: usize) !usize {
    const wanted = if (gib) |g| b: {
        if (!std.math.isFinite(g) or g < 0 or g >= 1 << 32) return error.BadCacheBudget;
        break :b @as(usize, @intFromFloat(g * (1 << 30)));
    } else @min(room, 1 << 30);
    if (slots == 0) return 0;
    if (wanted > room) return error.GlmOcrPromptCacheDoesNotFit;
    return wanted;
}

test "GLM-OCR native admission: cache budgets are finite and fit after planned lanes" {
    try std.testing.expectEqual(@as(usize, 1 << 30), try cacheBudget(null, 4 << 30, 8));
    try std.testing.expectEqual(@as(usize, 123), try cacheBudget(null, 123, 8));
    try std.testing.expectEqual(@as(usize, 0), try cacheBudget(1, 0, 0));
    try std.testing.expectEqual(@as(usize, 0), try cacheBudget(0, 0, 8));
    try std.testing.expectError(error.GlmOcrPromptCacheDoesNotFit, cacheBudget(1, 123, 8));
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64), 1 << 32 }) |g|
        try std.testing.expectError(error.BadCacheBudget, cacheBudget(g, 4 << 30, 8));
}
