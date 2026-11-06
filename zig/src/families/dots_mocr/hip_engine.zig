//! Stable ownership of one HIP model, its ordered scratch, staging buffers and preallocated lane cache bank.
const std = @import("std");
const hip = @import("hip");
const cfg = @import("config.zig");
const image = @import("image.zig");
const weights = @import("hip_weights.zig");
const graph = @import("hip_graph.zig");
const text = @import("hip_text.zig");
pub const plan = @import("hip_plan.zig");
const Tensor = hip.ops.Tensor;

pub const Slot = struct {
    cache: text.Cache,
    features: hip.DeviceBuffer,
    logits: hip.DeviceBuffer,
    active: bool = false,

    fn deinit(s: *Slot) void {
        s.logits.free();
        s.features.free();
        s.cache.deinit();
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    c: cfg.Config,
    p: plan.Plan,
    processor: ?image.Processor,
    driver: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    lib: hip.Launcher,
    blas: hip.blas.Blas,
    w: weights.Weights,
    arena: hip.Arena,
    ids: hip.DeviceBuffer,
    positions: hip.DeviceBuffer,
    pixels: hip.DeviceBuffer,
    host_ids: hip.HostBuffer,
    host_positions: hip.HostBuffer,
    /// Absent without --vision: the plan stages no pixels and pinned memory refuses zero bytes.
    host_pixels: ?hip.HostBuffer,
    slots: []Slot,
    before: hip.Usage,

    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, device: c_int, options: plan.Options) !*Engine {
        const e = try a.create(Engine);
        errdefer a.destroy(e);
        e.gpa = a;
        e.io = io;
        e.c = try cfg.Config.read(a, io, dir);
        e.processor = if (options.image_tokens > 0) try image.Processor.read(a, io, dir, e.c) else null;
        e.before = hip.usage(false);
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, device);
        errdefer e.ctx.deinit();
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        e.lib = try hip.Launcher.load(&e.driver, .{}, hip.kernels.images);
        errdefer e.lib.unload();
        e.blas = try hip.blas.Blas.open(e.stream);
        errdefer e.blas.close();
        var layout = try weights.Layout.init(a, e.c);
        defer layout.deinit();
        const pool = try hip.admission.budget(a, io, e.ctx);
        var o = options;
        o.unified = pool.unified;
        e.p = try plan.make(e.c, @intCast(pool.room(0)), layout.bytes, o);
        e.w = try weights.Weights.load(a, &e.driver, io, dir, e.c, layout.bytes);
        errdefer e.w.deinit();
        e.arena = try hip.Arena.init(&e.driver, e.p.scratch_bytes);
        errdefer e.arena.deinit();
        e.ids = try hip.DeviceBuffer.alloc(&e.driver, e.p.ids_bytes);
        errdefer e.ids.free();
        e.positions = try hip.DeviceBuffer.alloc(&e.driver, e.p.positions_bytes);
        errdefer e.positions.free();
        e.pixels = try hip.DeviceBuffer.alloc(&e.driver, e.p.pixels_bytes);
        errdefer e.pixels.free();
        e.host_ids = try hip.HostBuffer.alloc(&e.driver, e.p.ids_bytes);
        errdefer e.host_ids.free();
        e.host_positions = try hip.HostBuffer.alloc(&e.driver, e.p.positions_bytes);
        errdefer e.host_positions.free();
        e.host_pixels = if (e.p.pixels_bytes > 0) try hip.HostBuffer.alloc(&e.driver, e.p.pixels_bytes) else null;
        errdefer if (e.host_pixels) |*h| h.free();
        e.slots = try a.alloc(Slot, e.p.streams);
        errdefer a.free(e.slots);
        var made: usize = 0;
        errdefer for (e.slots[0..made]) |*s| s.deinit();
        for (e.slots) |*s| {
            var cache = try text.Cache.init(&e.driver, e.c, e.p.window);
            errdefer cache.deinit();
            var features = try hip.DeviceBuffer.alloc(&e.driver, e.p.feature_bytes);
            errdefer features.free();
            const logits = try hip.DeviceBuffer.alloc(&e.driver, @as(usize, e.c.vocab) * 2);
            s.* = .{ .cache = cache, .features = features, .logits = logits };
            made += 1;
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.ctx.makeCurrent() catch {};
        e.stream.synchronize() catch {};
        for (e.slots) |*s| s.deinit();
        e.gpa.free(e.slots);
        if (e.host_pixels) |*h| h.free();
        e.host_positions.free();
        e.host_ids.free();
        e.pixels.free();
        e.positions.free();
        e.ids.free();
        e.arena.deinit();
        e.w.deinit();
        e.blas.close();
        e.lib.unload();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn room(e: *Engine) !u64 {
        const pool = try hip.admission.budget(e.gpa, e.io, e.ctx);
        const now = hip.usage(false);
        const held = now.device - e.before.device + if (pool.unified) now.host - e.before.host else 0;
        return pool.room(held);
    }

    pub fn take(e: *Engine) !usize {
        for (e.slots, 0..) |*s, i| if (!s.active) {
            s.active = true;
            return i;
        };
        return error.NoFreeLane;
    }

    pub fn release(e: *Engine, slot: usize) void {
        e.stream.synchronize() catch {};
        e.slots[slot].active = false;
    }

    pub fn graph_(e: *Engine, window: bool) graph.Graph {
        return .{ .c = e.c, .w = &e.w, .blas = &e.blas, .ops = .{ .l = &e.lib, .bf16 = true, .stream = e.stream.handle, .arena = &e.arena, .window = window } };
    }

    pub fn forward(e: *Engine, slot: usize, tokens: []const u32, start: usize, spans: []const text.ImageSpan, window: bool, all_logits: bool) !Tensor {
        if (tokens.len == 0 or tokens.len > e.p.prompt_rows or start >= e.p.window or tokens.len > e.p.window - start)
            return error.DotsContext;
        try e.stream.synchronize();
        for (tokens, 0..) |id, i| {
            if (id >= e.c.vocab) return error.BadToken;
            e.host_ids.slice(u32)[i] = id;
            e.host_positions.slice(i32)[i] = @intCast(start + i);
        }
        e.arena.reset();
        try hip.raw.upload(e.ids, 0, e.host_ids.bytes[0 .. tokens.len * 4], e.stream.handle);
        try hip.raw.upload(e.positions, 0, e.host_positions.bytes[0 .. tokens.len * 4], e.stream.handle);
        const g = e.graph_(window);
        const input = try text.embed(g, e.ids.base(), tokens.len, spans);
        return (try text.forward(g, input, e.positions.base(), tokens.len, start, &e.slots[slot].cache, all_logits)).logits;
    }

    pub fn encode(e: *Engine, slot: usize, pixels: []const f32, gh: u32, gw: u32, feature_at: usize) !Tensor {
        const rows = @as(usize, gh) * gw;
        const tokens = rows / 4;
        const host_pixels = e.host_pixels orelse return error.DotsImageBudget;
        if (e.processor == null or rows == 0 or rows > e.p.image_rows or pixels.len != rows * 588 or
            feature_at > e.p.image_tokens or tokens > e.p.image_tokens - feature_at) return error.DotsImageBudget;
        try e.stream.synchronize();
        try image.fillPositions(e.host_positions.slice([2]u32)[0..rows], gh, gw, e.c.vision.merge);
        const bytes = std.mem.sliceAsBytes(pixels);
        @memcpy(host_pixels.bytes[0..bytes.len], bytes);
        e.arena.reset();
        try hip.raw.upload(e.pixels, 0, host_pixels.bytes[0..bytes.len], e.stream.handle);
        try hip.raw.upload(e.positions, 0, e.host_positions.bytes[0 .. rows * 8], e.stream.handle);
        const out = try e.graph_(false).vision(.{ .ptr = e.pixels.base(), .kind = .f32 }, e.positions.base(), rows);
        const offset = feature_at * e.c.hidden * 2;
        try hip.raw.copy(e.slots[slot].features, offset, out.ptr, tokens * e.c.hidden * 2, e.stream.handle);
        try e.stream.synchronize();
        return .{ .ptr = try e.slots[slot].features.address(offset), .kind = .bf16 };
    }
};
