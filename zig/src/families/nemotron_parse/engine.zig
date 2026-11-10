//! Owned single-rank HIP resources; the shared lane core owns scheduling and request lifetime.
const std = @import("std");
const hip = @import("hip");
const c = @import("config.zig");
const Weights = @import("weights.zig").Weights;
const Compute = @import("compute.zig").Compute;

pub const scratch_bytes = 1 << 30;
pub const weight_bytes = 1261601004;
pub const max_image_rows = 3329;
pub const max_raw_bytes = 13312 * 768 * 4;
pub const prefix_bytes = max_image_rows * c.hidden * 2;

pub fn stateBytes(window: usize) usize {
    return window * (c.hidden + c.layers * 2 * c.kv_heads * c.head_dim) * 2;
}

pub const Engine = struct {
    a: std.mem.Allocator,
    io: std.Io,
    r: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    launcher: hip.Launcher,
    kernels: hip.vision.Kernels,
    weights: Weights,
    scratch: hip.Arena,
    window: usize,
    streams: usize,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, dev: hip.Device, window: usize, asked: usize, fixed: bool) !*Engine {
        if (window == 0 or window > c.context or asked == 0 or asked > 64) return error.BadParseAdmission;
        const e = try a.create(Engine);
        errdefer a.destroy(e);
        e.a = a;
        e.io = io;
        e.window = window;
        e.r = try hip.Runtime.open();
        errdefer e.r.close();
        e.ctx = try hip.Context.init(&e.r, dev.index);
        errdefer e.ctx.deinit();
        _ = try hip.Device.kernelCaps(&e.r, dev.index);
        const pool = try hip.admission.budget(a, io, e.ctx);
        const room = pool.room(hip.usage(false).device);
        const base = weight_bytes + scratch_bytes + max_raw_bytes;
        const per = stateBytes(window) + prefix_bytes;
        if (room < base + per) return error.ParseDoesNotFit;
        e.streams = @min(asked, (room - base) / per);
        if (fixed and e.streams < asked) return error.ParseDoesNotFit;
        e.stream = try hip.Stream.init(&e.r);
        errdefer e.stream.deinit();
        e.launcher = try hip.Launcher.load(&e.r, dev.policy.choices(), hip.kernels.images);
        errdefer e.launcher.unload();
        // Every row uses the same attention tile across prompt chunks and shared verification widths.
        e.launcher.wide = false;
        e.kernels = try hip.vision.Kernels.load(e.launcher.mods[0]);
        e.weights = try Weights.load(a, &e.r, io, dir);
        errdefer e.weights.deinit();
        e.scratch = try hip.Arena.init(&e.r, scratch_bytes);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.scratch.deinit();
        e.weights.deinit();
        e.launcher.unload();
        e.stream.deinit();
        e.ctx.deinit();
        e.r.close();
        const a = e.a;
        a.destroy(e);
    }

    pub fn compute(e: *Engine) Compute {
        return .{ .ops = .{ .l = &e.launcher, .stream = e.stream.handle, .bf16 = true, .arena = &e.scratch }, .kernels = e.kernels, .stream = e.stream, .weights = &e.weights };
    }

    /// Refuse new lane storage if another GPU user consumed the unallocated part of our initial plan.
    pub fn laneRoom(e: *Engine) !void {
        const pool = try hip.admission.budget(e.a, e.io, e.ctx);
        if (pool.room(hip.usage(false).device) < stateBytes(e.window) + prefix_bytes + max_raw_bytes) return error.LaneMemory;
    }
};
